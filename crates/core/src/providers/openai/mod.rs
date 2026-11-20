// SPDX-License-Identifier: AGPL-3.0-only

//! OpenAI Chat Completions API: request building.

use std::fmt;

use serde_json::{Map, Value, json};

use super::ProviderError;
use crate::domain::chat::{ChatRequest, ContentPart, Message, Role, ToolChoice};
use crate::ports::HttpRequest;

/// OpenAI's public API endpoint.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// Where and how to reach OpenAI.
#[derive(Clone)]
pub struct OpenAiConfig {
    /// Base URL without a trailing path, for example [`DEFAULT_BASE_URL`].
    pub base_url: String,
    /// API key sent as a bearer token.
    pub api_key: String,
}

impl fmt::Debug for OpenAiConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiConfig")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

/// What the request builder needs to know about the target model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelLimits {
    /// The model accepts tools only with `reasoning_effort: "none"`; from
    /// the catalog's `has_tool_reasoning_limit`.
    pub has_tool_reasoning_limit: bool,
}

/// Builds the HTTP request for `request`.
pub fn build_request(
    config: &OpenAiConfig,
    request: &ChatRequest,
    limits: ModelLimits,
    stream: bool,
) -> Result<HttpRequest, ProviderError> {
    let mut body = Map::new();
    body.insert("model".into(), json!(request.model));
    let mut messages = Vec::new();
    // OpenAI's docs: from o1 on, developer messages replace system
    // messages; older models convert developer to system.
    if let Some(system) = &request.system {
        messages.push(json!({ "role": "developer", "content": system }));
    }
    for m in &request.messages {
        append_message(&mut messages, m)?;
    }
    body.insert("messages".into(), Value::Array(messages));
    if !request.tools.is_empty() {
        let tools = request.tools.iter().map(|t| {
            let mut function = json!({ "name": t.name, "parameters": t.parameters });
            if let Some(description) = &t.description {
                function["description"] = json!(description);
            }
            json!({ "type": "function", "function": function })
        });
        body.insert("tools".into(), Value::Array(tools.collect()));
        if limits.has_tool_reasoning_limit {
            body.insert("reasoning_effort".into(), json!("none"));
        }
    }
    if let Some(choice) = &request.tool_choice {
        let choice = match choice {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::None => json!("none"),
            ToolChoice::Required => json!("required"),
            ToolChoice::Tool(name) => json!({ "type": "function", "function": { "name": name } }),
        };
        body.insert("tool_choice".into(), choice);
    }
    if let Some(max) = request.max_tokens {
        body.insert("max_completion_tokens".into(), json!(max));
    }
    if let Some(t) = request.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(p) = request.top_p {
        body.insert("top_p".into(), json!(p));
    }
    if !request.stop.is_empty() {
        body.insert("stop".into(), json!(request.stop));
    }
    if stream {
        body.insert("stream".into(), json!(true));
        body.insert("stream_options".into(), json!({ "include_usage": true }));
    }
    Ok(HttpRequest {
        url: format!(
            "{}/v1/chat/completions",
            config.base_url.trim_end_matches('/')
        ),
        headers: vec![
            ("authorization".into(), format!("Bearer {}", config.api_key)),
            ("content-type".into(), "application/json".into()),
        ],
        body: Value::Object(body).to_string().into_bytes(),
    })
}

// One neutral message can become several OpenAI messages: each tool
// result is its own `tool` message, and those must directly follow the
// assistant message that made the calls, so they come before user text.
fn append_message(out: &mut Vec<Value>, message: &Message) -> Result<(), ProviderError> {
    let mut texts = Vec::new();
    let mut calls = Vec::new();
    for part in &message.content {
        match (part, message.role) {
            (ContentPart::Text(text), _) => texts.push(text.as_str()),
            (ContentPart::ToolCall(call), Role::Assistant) => calls.push(json!({
                "id": call.id,
                "type": "function",
                "function": { "name": call.name, "arguments": call.arguments },
            })),
            (
                ContentPart::ToolResult {
                    call_id, content, ..
                },
                Role::User,
            ) => {
                out.push(json!({ "role": "tool", "tool_call_id": call_id, "content": content }));
            }
            (ContentPart::ToolCall(_), Role::User) => {
                return Err(ProviderError::InvalidRequest(
                    "a user message cannot contain a tool call".into(),
                ));
            }
            (ContentPart::ToolResult { .. }, Role::Assistant) => {
                return Err(ProviderError::InvalidRequest(
                    "an assistant message cannot contain a tool result".into(),
                ));
            }
        }
    }
    let text = (!texts.is_empty()).then(|| texts.concat());
    match message.role {
        Role::User => {
            if let Some(text) = text {
                out.push(json!({ "role": "user", "content": text }));
            }
        }
        Role::Assistant => {
            let mut assistant = json!({ "role": "assistant", "content": text });
            if !calls.is_empty() {
                assistant["tool_calls"] = Value::Array(calls);
            }
            out.push(assistant);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::{ToolCall, ToolDefinition};

    fn config() -> OpenAiConfig {
        OpenAiConfig {
            base_url: "https://openai.test/".into(),
            api_key: "test-key".into(),
        }
    }

    fn text(role: Role, text: &str) -> Message {
        Message {
            role,
            content: vec![ContentPart::Text(text.into())],
        }
    }

    fn request(messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            model: "gpt-6-luna".into(),
            system: None,
            messages,
            tools: Vec::new(),
            tool_choice: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            stream: false,
        }
    }

    fn body(http: &HttpRequest) -> Value {
        serde_json::from_slice(&http.body).unwrap()
    }

    #[test]
    fn builds_a_full_request() {
        let mut req = request(vec![
            text(Role::User, "Weather in Lima?"),
            Message {
                role: Role::Assistant,
                content: vec![ContentPart::ToolCall(ToolCall {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    arguments: r#"{"city":"Lima"}"#.into(),
                })],
            },
            Message {
                role: Role::User,
                content: vec![
                    ContentPart::Text("And tomorrow?".into()),
                    ContentPart::ToolResult {
                        call_id: "call_1".into(),
                        content: "18 C".into(),
                        is_error: false,
                    },
                ],
            },
        ]);
        req.system = Some("Be brief.".into());
        req.tools = vec![ToolDefinition {
            name: "get_weather".into(),
            description: Some("Current weather.".into()),
            parameters: json!({ "type": "object" }),
        }];
        req.tool_choice = Some(ToolChoice::Tool("get_weather".into()));
        req.max_tokens = Some(300);
        req.stop = vec!["END".into()];
        let http = build_request(&config(), &req, ModelLimits::default(), true).unwrap();
        assert_eq!(http.url, "https://openai.test/v1/chat/completions");
        assert!(
            http.headers
                .contains(&("authorization".into(), "Bearer test-key".into()))
        );
        assert_eq!(
            body(&http),
            json!({
                "model": "gpt-6-luna",
                "messages": [
                    { "role": "developer", "content": "Be brief." },
                    { "role": "user", "content": "Weather in Lima?" },
                    { "role": "assistant", "content": null, "tool_calls": [
                        { "id": "call_1", "type": "function", "function": { "name": "get_weather", "arguments": "{\"city\":\"Lima\"}" } },
                    ] },
                    { "role": "tool", "tool_call_id": "call_1", "content": "18 C" },
                    { "role": "user", "content": "And tomorrow?" },
                ],
                "tools": [{ "type": "function", "function": {
                    "name": "get_weather", "description": "Current weather.", "parameters": { "type": "object" },
                } }],
                "tool_choice": { "type": "function", "function": { "name": "get_weather" } },
                "max_completion_tokens": 300,
                "stop": ["END"],
                "stream": true,
                "stream_options": { "include_usage": true },
            })
        );
        assert!(!format!("{http:?}{:?}", config()).contains("test-key"));
    }

    #[test]
    fn reasoning_is_turned_off_only_for_tools_on_limited_models() {
        let limited = ModelLimits {
            has_tool_reasoning_limit: true,
        };
        let mut with_tools = request(vec![text(Role::User, "hi")]);
        with_tools.tools = vec![ToolDefinition {
            name: "t".into(),
            description: None,
            parameters: json!({ "type": "object" }),
        }];
        let effort = |req: &ChatRequest, limits| {
            body(&build_request(&config(), req, limits, false).unwrap())["reasoning_effort"].clone()
        };
        assert_eq!(effort(&with_tools, limited), json!("none"));
        assert_eq!(effort(&with_tools, ModelLimits::default()), Value::Null);
        assert_eq!(
            effort(&request(vec![text(Role::User, "hi")]), limited),
            Value::Null
        );
    }

    #[test]
    fn plain_assistant_text_and_no_optional_fields() {
        let req = request(vec![text(Role::User, "hi"), text(Role::Assistant, "hello")]);
        let sent = body(&build_request(&config(), &req, ModelLimits::default(), false).unwrap());
        assert_eq!(
            sent,
            json!({
                "model": "gpt-6-luna",
                "messages": [{ "role": "user", "content": "hi" }, { "role": "assistant", "content": "hello" }],
            })
        );
    }

    #[test]
    fn rejects_misplaced_tool_parts() {
        let call = ContentPart::ToolCall(ToolCall {
            id: "c".into(),
            name: "t".into(),
            arguments: "{}".into(),
        });
        let result = ContentPart::ToolResult {
            call_id: "c".into(),
            content: String::new(),
            is_error: false,
        };
        for m in [
            Message {
                role: Role::User,
                content: vec![call],
            },
            Message {
                role: Role::Assistant,
                content: vec![result],
            },
        ] {
            let built = build_request(&config(), &request(vec![m]), ModelLimits::default(), false);
            assert!(
                matches!(built, Err(ProviderError::InvalidRequest(_))),
                "{built:?}"
            );
        }
    }
}
