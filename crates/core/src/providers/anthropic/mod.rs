// SPDX-License-Identifier: AGPL-3.0-only

//! Anthropic Messages API: request building and response parsing, whole
//! or streamed.

use std::fmt;

use serde_json::{Map, Value, json};

use super::ProviderError;
use crate::domain::chat::{
    ChatRequest, ChatResponse, ContentPart, FinishReason, Role, ToolCall, ToolChoice, Usage,
};
use crate::ports::{HttpClient, HttpRequest, read_to_end};

mod stream;

pub use stream::{AnthropicStream, stream};

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

/// Maps Anthropic's `stop_reason` to the neutral form.
pub(crate) fn finish_reason(stop_reason: &str) -> FinishReason {
    match stop_reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "refusal" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_string()),
    }
}

/// Reads Anthropic's `usage` object. Writes are split by cache lifetime
/// from `cache_creation`; any part of the total it does not account for,
/// or the whole total when it is absent, counts as short-lived.
pub(crate) fn usage(value: &Value) -> Usage {
    let count = |v: &Value, key: &str| v[key].as_u64().unwrap_or(0);
    let written = count(value, "cache_creation_input_tokens");
    let breakdown = &value["cache_creation"];
    let long = count(breakdown, "ephemeral_1h_input_tokens");
    let short = count(breakdown, "ephemeral_5m_input_tokens");
    Usage {
        input_tokens: count(value, "input_tokens"),
        output_tokens: count(value, "output_tokens"),
        cache_read_tokens: count(value, "cache_read_input_tokens"),
        cache_write_short_tokens: short.max(written.saturating_sub(long)),
        cache_write_long_tokens: long,
    }
}

/// Turns an HTTP status and body into a response or a typed error.
pub fn parse_response(status: u16, body: &[u8]) -> Result<ChatResponse, ProviderError> {
    if !(200..300).contains(&status) {
        return Err(status_error(status, body));
    }
    let malformed = |what: &str| ProviderError::Malformed(format!("anthropic response {what}"));
    let value: Value =
        serde_json::from_slice(body).map_err(|e| ProviderError::Malformed(e.to_string()))?;
    let text_field = |key: &str| value[key].as_str().map(str::to_string);
    let blocks = value["content"]
        .as_array()
        .ok_or_else(|| malformed("has no content array"))?;
    let mut content = Vec::new();
    for block in blocks {
        match block["type"].as_str() {
            Some("text") => {
                let text = block["text"]
                    .as_str()
                    .ok_or_else(|| malformed("has a text block without text"))?;
                content.push(ContentPart::Text(text.to_string()));
            }
            Some("tool_use") => {
                let (Some(id), Some(name)) = (block["id"].as_str(), block["name"].as_str()) else {
                    return Err(malformed("has a tool_use block without id or name"));
                };
                content.push(ContentPart::ToolCall(ToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments: block["input"].to_string(),
                }));
            }
            // Thinking and server-tool blocks have no neutral form yet.
            _ => {}
        }
    }
    Ok(ChatResponse {
        id: text_field("id").ok_or_else(|| malformed("has no id"))?,
        model: text_field("model").ok_or_else(|| malformed("has no model"))?,
        content,
        finish_reason: value["stop_reason"].as_str().map(finish_reason),
        usage: usage(&value["usage"]),
    })
}

/// Reads Anthropic's error body; an unreadable one still yields the status.
pub(crate) fn status_error(status: u16, body: &[u8]) -> ProviderError {
    let value: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    let error = &value["error"];
    ProviderError::Status {
        status,
        error_type: error["type"].as_str().map(str::to_string),
        message: error["message"]
            .as_str()
            .unwrap_or("error body was not readable")
            .to_string(),
    }
}

/// Sends a non-streamed request and returns the whole response.
pub async fn complete<H: HttpClient>(
    client: &H,
    config: &AnthropicConfig,
    request: &ChatRequest,
    default_max_tokens: u32,
) -> Result<ChatResponse, ProviderError> {
    let http = build_request(config, request, default_max_tokens, false)?;
    let response = client.send(http).await?;
    let body = read_to_end(response.body).await?;
    parse_response(response.status, &body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::{Message, ToolDefinition};
    use crate::ports::HttpError;
    use crate::ports::http::fake::FakeHttpClient;
    use crate::test_support::block_on;

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

    fn call(
        fixture: &str,
        status: u16,
        chunk: usize,
    ) -> (Result<ChatResponse, ProviderError>, Value) {
        let client = FakeHttpClient::new(status, fixture, chunk);
        let result = block_on(complete(&client, &config(), &request(vec![user("hi")]), 64));
        let sent = serde_json::from_slice(&client.requests()[0].body).unwrap();
        (result, sent)
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

    #[test]
    fn plain_chat() {
        let (result, sent) = call(
            include_str!("../../../fixtures/anthropic/chat.json"),
            200,
            7,
        );
        let response = result.unwrap();
        assert_eq!(response.id, "msg_fixture01");
        assert_eq!(response.model, "claude-haiku-5-5");
        assert_eq!(
            response.content,
            vec![ContentPart::Text("Hello from the fixture.".into())]
        );
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            response.usage,
            Usage {
                input_tokens: 34,
                output_tokens: 11,
                ..Usage::default()
            }
        );
        assert_eq!(sent["max_tokens"], 64);
        assert_eq!(sent.get("stream"), None);
    }

    #[test]
    fn tool_call() {
        let (result, _) = call(
            include_str!("../../../fixtures/anthropic/tool.json"),
            200,
            64,
        );
        let response = result.unwrap();
        assert_eq!(
            response.content,
            vec![ContentPart::ToolCall(ToolCall {
                id: "toolu_fixture02".into(),
                name: "get_weather".into(),
                arguments: r#"{"city":"Lima"}"#.into(),
            })]
        );
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(
            response.usage,
            Usage {
                input_tokens: 380,
                output_tokens: 50,
                ..Usage::default()
            }
        );
    }

    #[test]
    fn cache_write_and_read_counts() {
        let (write, _) = call(
            include_str!("../../../fixtures/anthropic/cache-write.json"),
            200,
            64,
        );
        let (read, _) = call(
            include_str!("../../../fixtures/anthropic/cache-read.json"),
            200,
            64,
        );
        let base = Usage {
            input_tokens: 32,
            output_tokens: 11,
            ..Usage::default()
        };
        assert_eq!(
            write.unwrap().usage,
            Usage {
                cache_write_long_tokens: 5403,
                ..base
            }
        );
        assert_eq!(
            read.unwrap().usage,
            Usage {
                cache_read_tokens: 5403,
                ..base
            }
        );
    }

    #[test]
    fn writes_without_a_breakdown_count_as_short() {
        let reported = json!({ "input_tokens": 1, "cache_creation_input_tokens": 10 });
        assert_eq!(usage(&reported).cache_write_short_tokens, 10);
        let partial = json!({ "cache_creation_input_tokens": 10, "cache_creation": { "ephemeral_1h_input_tokens": 4 } });
        assert_eq!(
            (
                usage(&partial).cache_write_short_tokens,
                usage(&partial).cache_write_long_tokens
            ),
            (6, 4)
        );
    }

    #[test]
    fn provider_error() {
        let (result, _) = call(
            include_str!("../../../fixtures/anthropic/error-404.json"),
            404,
            64,
        );
        assert_eq!(
            result,
            Err(ProviderError::Status {
                status: 404,
                error_type: Some("not_found_error".into()),
                message: "model: lleme-fixture-no-such-model".into(),
            })
        );
        let (unreadable, _) = call("<html>bad gateway</html>", 502, 64);
        assert!(matches!(
            unreadable,
            Err(ProviderError::Status {
                status: 502,
                error_type: None,
                ..
            })
        ));
    }

    #[test]
    fn malformed_response() {
        let truncated = &include_str!("../../../fixtures/anthropic/chat.json")[..80];
        assert!(matches!(
            call(truncated, 200, 64).0,
            Err(ProviderError::Malformed(_))
        ));
        assert!(matches!(
            call(r#"{"id":"m","model":"x"}"#, 200, 64).0,
            Err(ProviderError::Malformed(_))
        ));
    }

    #[test]
    fn body_that_breaks_off() {
        let client = FakeHttpClient::new(
            200,
            include_str!("../../../fixtures/anthropic/chat.json"),
            16,
        )
        .failing_after(2);
        let result = block_on(complete(&client, &config(), &request(vec![user("hi")]), 64));
        assert_eq!(
            result,
            Err(ProviderError::Http(HttpError::Body(
                "connection reset".into()
            )))
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn complete_future_is_send_on_native() {
        fn assert_send<T: Send>(_: &T) {}
        let client = FakeHttpClient::new(200, "", 1);
        let (cfg, req) = (config(), request(Vec::new()));
        let future = complete(&client, &cfg, &req, 1);
        assert_send(&future);
    }
}
