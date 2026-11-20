// SPDX-License-Identifier: AGPL-3.0-only

//! Anthropic Messages API: request building.

use std::fmt;

use serde_json::{Map, Value, json};

use super::ProviderError;
use crate::domain::chat::{ChatRequest, ContentPart, Role, ToolCall, ToolChoice};
use crate::ports::HttpRequest;

/// Anthropic's public API endpoint.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// API version sent in `anthropic-version`; fixtures were recorded with it.
const API_VERSION: &str = "2026-10-21";

/// Where and how to reach Anthropic.
#[derive(Clone)]
pub struct AnthropicConfig {
    /// Base URL without a trailing path, for example [`DEFAULT_BASE_URL`].
    pub base_url: String,
    /// API key sent in `x-api-key`.
    pub api_key: String,
}

impl fmt::Debug for AnthropicConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicConfig")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

/// Builds the HTTP request for `request`. Anthropic requires `max_tokens`,
/// so `default_max_tokens` is used when the request has none.
pub fn build_request(
    config: &AnthropicConfig,
    request: &ChatRequest,
    default_max_tokens: u32,
    stream: bool,
) -> Result<HttpRequest, ProviderError> {
    let mut body = Map::new();
    body.insert("model".into(), json!(request.model));
    body.insert(
        "max_tokens".into(),
        json!(request.max_tokens.unwrap_or(default_max_tokens)),
    );
    if let Some(system) = &request.system {
        body.insert("system".into(), json!(system));
    }
    let messages = request
        .messages
        .iter()
        .map(message)
        .collect::<Result<Vec<_>, _>>()?;
    body.insert("messages".into(), Value::Array(messages));
    if !request.tools.is_empty() {
        let tools = request.tools.iter().map(|t| {
            let mut tool = json!({ "name": t.name, "input_schema": t.parameters });
            if let Some(description) = &t.description {
                tool["description"] = json!(description);
            }
            tool
        });
        body.insert("tools".into(), Value::Array(tools.collect()));
    }
    if let Some(choice) = &request.tool_choice {
        let choice = match choice {
            ToolChoice::Auto => json!({ "type": "auto" }),
            ToolChoice::None => json!({ "type": "none" }),
            ToolChoice::Required => json!({ "type": "any" }),
            ToolChoice::Tool(name) => json!({ "type": "tool", "name": name }),
        };
        body.insert("tool_choice".into(), choice);
    }
    if let Some(t) = request.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(p) = request.top_p {
        body.insert("top_p".into(), json!(p));
    }
    if !request.stop.is_empty() {
        body.insert("stop_sequences".into(), json!(request.stop));
    }
    if stream {
        body.insert("stream".into(), json!(true));
    }
    Ok(HttpRequest {
        url: format!("{}/v1/messages", config.base_url.trim_end_matches('/')),
        headers: vec![
            ("x-api-key".into(), config.api_key.clone()),
            ("anthropic-version".into(), API_VERSION.into()),
            ("content-type".into(), "application/json".into()),
        ],
        body: Value::Object(body).to_string().into_bytes(),
    })
}

fn message(message: &crate::domain::chat::Message) -> Result<Value, ProviderError> {
    let role = match message.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    let mut blocks = Vec::with_capacity(message.content.len());
    for part in &message.content {
        blocks.push(match (part, message.role) {
            (ContentPart::Text(text), _) => json!({ "type": "text", "text": text }),
            (ContentPart::ToolCall(call), Role::Assistant) => {
                json!({ "type": "tool_use", "id": call.id, "name": call.name, "input": tool_input(call)? })
            }
            (ContentPart::ToolResult { call_id, content, is_error }, Role::User) => {
                let mut block = json!({ "type": "tool_result", "tool_use_id": call_id, "content": content });
                if *is_error {
                    block["is_error"] = json!(true);
                }
                block
            }
            (ContentPart::ToolCall(_), Role::User) => {
                return Err(ProviderError::InvalidRequest("a user message cannot contain a tool call".into()));
            }
            (ContentPart::ToolResult { .. }, Role::Assistant) => {
                return Err(ProviderError::InvalidRequest("an assistant message cannot contain a tool result".into()));
            }
        });
    }
    Ok(json!({ "role": role, "content": blocks }))
}

// Anthropic takes tool input as a JSON object; the neutral form keeps text.
fn tool_input(call: &ToolCall) -> Result<Value, ProviderError> {
    if call.arguments.trim().is_empty() {
        return Ok(json!({}));
    }
    match serde_json::from_str::<Value>(&call.arguments) {
        Ok(value @ Value::Object(_)) => Ok(value),
        _ => Err(ProviderError::InvalidRequest(format!(
            "arguments of tool call {} are not a JSON object",
            call.id
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::{Message, ToolDefinition};

    fn config() -> AnthropicConfig {
        AnthropicConfig {
            base_url: "https://anthropic.test/".into(),
            api_key: "test-key".into(),
        }
    }

    fn user(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentPart::Text(text.into())],
        }
    }

    fn request(messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            model: "claude-haiku-5-5".into(),
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

    #[test]
    fn builds_a_full_request() {
        let mut req = request(vec![
            user("Weather in Lima?"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentPart::Text("Checking.".into()),
                    ContentPart::ToolCall(ToolCall {
                        id: "c1".into(),
                        name: "get_weather".into(),
                        arguments: r#"{"city":"Lima"}"#.into(),
                    }),
                ],
            },
            Message {
                role: Role::User,
                content: vec![ContentPart::ToolResult {
                    call_id: "c1".into(),
                    content: "18 C".into(),
                    is_error: true,
                }],
            },
        ]);
        req.system = Some("Be brief.".into());
        req.tools = vec![ToolDefinition {
            name: "get_weather".into(),
            description: Some("Current weather.".into()),
            parameters: json!({ "type": "object" }),
        }];
        req.tool_choice = Some(ToolChoice::Required);
        req.max_tokens = Some(300);
        req.temperature = Some(0.5);
        req.stop = vec!["END".into()];
        let http = build_request(&config(), &req, 64, true).unwrap();
        assert_eq!(http.url, "https://anthropic.test/v1/messages");
        assert!(
            http.headers
                .contains(&("anthropic-version".into(), API_VERSION.into()))
        );
        assert!(
            http.headers
                .contains(&("x-api-key".into(), "test-key".into()))
        );
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        assert_eq!(
            body,
            json!({
                "model": "claude-haiku-5-5",
                "max_tokens": 300,
                "system": "Be brief.",
                "messages": [
                    { "role": "user", "content": [{ "type": "text", "text": "Weather in Lima?" }] },
                    { "role": "assistant", "content": [
                        { "type": "text", "text": "Checking." },
                        { "type": "tool_use", "id": "c1", "name": "get_weather", "input": { "city": "Lima" } },
                    ] },
                    { "role": "user", "content": [
                        { "type": "tool_result", "tool_use_id": "c1", "content": "18 C", "is_error": true },
                    ] },
                ],
                "tools": [{ "name": "get_weather", "description": "Current weather.", "input_schema": { "type": "object" } }],
                "tool_choice": { "type": "any" },
                "temperature": 0.5,
                "stop_sequences": ["END"],
                "stream": true,
            })
        );
        assert!(!format!("{http:?}{:?}", config()).contains("test-key"));
    }

    #[test]
    fn rejects_what_anthropic_cannot_express() {
        let bad_args = Message {
            role: Role::Assistant,
            content: vec![ContentPart::ToolCall(ToolCall {
                id: "c".into(),
                name: "t".into(),
                arguments: "[1]".into(),
            })],
        };
        let misplaced = Message {
            role: Role::Assistant,
            content: vec![ContentPart::ToolResult {
                call_id: "c".into(),
                content: String::new(),
                is_error: false,
            }],
        };
        for m in [bad_args, misplaced] {
            let result = build_request(&config(), &request(vec![m]), 64, false);
            assert!(
                matches!(result, Err(ProviderError::InvalidRequest(_))),
                "{result:?}"
            );
        }
    }
}
