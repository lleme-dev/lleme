// SPDX-License-Identifier: AGPL-3.0-only

//! Anthropic-format client requests read into the neutral form, for
//! sending to a provider with another format. Same-format requests are
//! forwarded as is and never pass through here.

use serde_json::Value;

use crate::domain::chat::{
    ChatRequest, ContentPart, Message, Role, ToolCall, ToolChoice, ToolDefinition,
};
use crate::providers::RequestError;

fn invalid(what: &str) -> RequestError {
    RequestError::Invalid(what.to_string())
}

/// Reads a Messages API request body.
pub fn parse_request(body: &[u8]) -> Result<ChatRequest, RequestError> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|e| RequestError::Invalid(format!("body is not JSON: {e}")))?;
    let model = value["model"]
        .as_str()
        .ok_or_else(|| invalid("model must be a string"))?;
    let system = match &value["system"] {
        Value::Null => None,
        system => Some(text_of(system)?),
    };
    let mut messages: Vec<Message> = Vec::new();
    for m in value["messages"]
        .as_array()
        .ok_or_else(|| invalid("messages must be an array"))?
    {
        let role = match m["role"].as_str() {
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            Some(other) => {
                return Err(RequestError::Invalid(format!(
                    "unknown message role {other:?}"
                )));
            }
            None => return Err(invalid("every message needs a role")),
        };
        let parts = parts(&m["content"], role)?;
        match messages.last_mut() {
            Some(last) if last.role == role => last.content.extend(parts),
            _ => messages.push(Message {
                role,
                content: parts,
            }),
        }
    }
    Ok(ChatRequest {
        model: model.to_string(),
        system,
        messages,
        tools: tools(&value["tools"])?,
        tool_choice: tool_choice(&value["tool_choice"])?,
        max_tokens: match &value["max_tokens"] {
            Value::Null => None,
            v => Some(
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| invalid("max_tokens must be a non-negative integer"))?,
            ),
        },
        temperature: optional_f64(&value, "temperature")?,
        top_p: optional_f64(&value, "top_p")?,
        stop: match &value["stop_sequences"] {
            Value::Null => Vec::new(),
            Value::Array(items) => items
                .iter()
                .map(|s| {
                    s.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| invalid("stop_sequences must hold strings"))
                })
                .collect::<Result<_, _>>()?,
            _ => return Err(invalid("stop_sequences must be an array")),
        },
        stream: value["stream"].as_bool().unwrap_or(false),
    })
}

// A string, or blocks that are all text (system prompts, tool results).
fn text_of(content: &Value) -> Result<String, RequestError> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(blocks) => blocks
            .iter()
            .map(|block| match block["type"].as_str() {
                Some("text") => block["text"]
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| invalid("text blocks need text")),
                Some(other) => Err(RequestError::Unsupported(format!("{other} blocks"))),
                None => Err(invalid("content blocks need a type")),
            })
            .collect(),
        _ => Err(invalid("content must be a string or an array of blocks")),
    }
}

fn parts(content: &Value, role: Role) -> Result<Vec<ContentPart>, RequestError> {
    let Value::Array(blocks) = content else {
        return Ok(vec![ContentPart::Text(text_of(content)?)]);
    };
    let mut parts = Vec::new();
    for block in blocks {
        match (block["type"].as_str(), role) {
            (Some("text"), _) => {
                let text = block["text"]
                    .as_str()
                    .ok_or_else(|| invalid("text blocks need text"))?;
                parts.push(ContentPart::Text(text.to_string()));
            }
            (Some("tool_use"), Role::Assistant) => {
                let (Some(id), Some(name)) = (block["id"].as_str(), block["name"].as_str()) else {
                    return Err(invalid("tool_use blocks need an id and a name"));
                };
                parts.push(ContentPart::ToolCall(ToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments: block["input"].to_string(),
                }));
            }
            (Some("tool_result"), Role::User) => {
                let call_id = block["tool_use_id"]
                    .as_str()
                    .ok_or_else(|| invalid("tool_result blocks need tool_use_id"))?;
                let content = match &block["content"] {
                    Value::Null => String::new(),
                    content => text_of(content)?,
                };
                let is_error = block["is_error"].as_bool().unwrap_or(false);
                parts.push(ContentPart::ToolResult {
                    call_id: call_id.to_string(),
                    content,
                    is_error,
                });
            }
            // Signed reasoning means nothing to another provider.
            (Some("thinking" | "redacted_thinking"), Role::Assistant) => {}
            (Some(kind @ ("tool_use" | "tool_result" | "thinking" | "redacted_thinking")), _) => {
                return Err(RequestError::Invalid(format!(
                    "{kind} blocks are not allowed in this role"
                )));
            }
            (Some(other), _) => return Err(RequestError::Unsupported(format!("{other} blocks"))),
            (None, _) => return Err(invalid("content blocks need a type")),
        }
    }
    Ok(parts)
}

fn tools(value: &Value) -> Result<Vec<ToolDefinition>, RequestError> {
    let Value::Array(items) = value else {
        return if value.is_null() {
            Ok(Vec::new())
        } else {
            Err(invalid("tools must be an array"))
        };
    };
    items
        .iter()
        .map(|tool| {
            if !matches!(tool["type"].as_str(), None | Some("custom")) {
                return Err(RequestError::Unsupported("Anthropic server tools".into()));
            }
            let name = tool["name"]
                .as_str()
                .ok_or_else(|| invalid("tools need a name"))?;
            if tool["input_schema"].is_null() {
                return Err(invalid("tools need an input_schema"));
            }
            Ok(ToolDefinition {
                name: name.to_string(),
                description: tool["description"].as_str().map(str::to_string),
                parameters: tool["input_schema"].clone(),
            })
        })
        .collect()
}

fn tool_choice(value: &Value) -> Result<Option<ToolChoice>, RequestError> {
    if value.is_null() {
        return Ok(None);
    }
    Ok(Some(match value["type"].as_str() {
        Some("auto") => ToolChoice::Auto,
        Some("any") => ToolChoice::Required,
        Some("none") => ToolChoice::None,
        Some("tool") => ToolChoice::Tool(
            value["name"]
                .as_str()
                .ok_or_else(|| invalid("tool_choice needs a name"))?
                .to_string(),
        ),
        _ => return Err(invalid("tool_choice needs a known type")),
    }))
}

fn optional_f64(value: &Value, key: &str) -> Result<Option<f64>, RequestError> {
    match &value[key] {
        Value::Null => Ok(None),
        v => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| RequestError::Invalid(format!("{key} must be a number"))),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn parse(value: Value) -> Result<ChatRequest, RequestError> {
        parse_request(value.to_string().as_bytes())
    }

    #[test]
    fn reads_a_full_conversation() {
        let request = parse(json!({
            "model": "claude-haiku-5-5",
            "max_tokens": 300,
            "system": [{ "type": "text", "text": "Be brief.", "cache_control": { "type": "ephemeral" } }],
            "messages": [
                { "role": "user", "content": "Weather in Lima?" },
                { "role": "assistant", "content": [
                    { "type": "thinking", "thinking": "...", "signature": "sig" },
                    { "type": "text", "text": "Checking." },
                    { "type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": { "city": "Lima" } },
                ] },
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "toolu_1", "content": [{ "type": "text", "text": "18 C" }], "is_error": true },
                ] },
                { "role": "user", "content": "And tomorrow?" },
            ],
            "tools": [{ "name": "get_weather", "description": "Weather.", "input_schema": { "type": "object" } }],
            "tool_choice": { "type": "any" },
            "temperature": 0.2,
            "stop_sequences": ["END"],
            "stream": true,
            "top_k": 5,
        }))
        .unwrap();
        assert_eq!(request.system.as_deref(), Some("Be brief."));
        assert_eq!(
            request.messages,
            [
                Message {
                    role: Role::User,
                    content: vec![ContentPart::Text("Weather in Lima?".into())]
                },
                Message {
                    role: Role::Assistant,
                    content: vec![
                        ContentPart::Text("Checking.".into()),
                        ContentPart::ToolCall(ToolCall {
                            id: "toolu_1".into(),
                            name: "get_weather".into(),
                            arguments: r#"{"city":"Lima"}"#.into(),
                        }),
                    ],
                },
                Message {
                    role: Role::User,
                    content: vec![
                        ContentPart::ToolResult {
                            call_id: "toolu_1".into(),
                            content: "18 C".into(),
                            is_error: true
                        },
                        ContentPart::Text("And tomorrow?".into()),
                    ],
                },
            ]
        );
        assert_eq!(request.tools[0].parameters, json!({ "type": "object" }));
        assert_eq!(request.tool_choice, Some(ToolChoice::Required));
        assert_eq!(
            (request.max_tokens, request.stop.clone(), request.stream),
            (Some(300), vec!["END".to_string()], true)
        );
    }

    #[test]
    fn minimal_request() {
        let request =
            parse(json!({ "model": "m", "messages": [{ "role": "user", "content": "hi" }] }))
                .unwrap();
        assert_eq!(
            (
                request.system,
                request.max_tokens,
                request.tool_choice,
                request.stream
            ),
            (None, None, None, false)
        );
    }

    #[test]
    fn rejects_invalid_requests() {
        let cases = [
            json!({ "messages": [] }),
            json!({ "model": "m", "messages": [{ "role": "system", "content": "x" }] }),
            json!({ "model": "m", "messages": [{ "role": "user", "content": [{ "type": "tool_use", "id": "a", "name": "b", "input": {} }] }] }),
            json!({ "model": "m", "messages": [{ "role": "assistant", "content": [{ "type": "tool_result", "tool_use_id": "a" }] }] }),
            json!({ "model": "m", "messages": [], "tools": [{ "name": "t" }] }),
            json!({ "model": "m", "messages": [], "tool_choice": { "type": "tool" } }),
            json!({ "model": "m", "messages": [], "max_tokens": 1.5 }),
            json!({ "model": "m", "messages": [], "stop_sequences": "END" }),
        ];
        for case in cases {
            assert!(
                matches!(parse(case.clone()), Err(RequestError::Invalid(_))),
                "{case}"
            );
        }
    }

    #[test]
    fn reports_what_cannot_be_translated_yet() {
        let image = json!({ "type": "image", "source": { "type": "url", "url": "https://example.test/a.png" } });
        let cases = [
            json!({ "model": "m", "messages": [{ "role": "user", "content": [image] }] }),
            json!({ "model": "m", "messages": [], "tools": [{ "type": "web_search_20260209", "name": "web_search" }] }),
        ];
        for case in cases {
            assert!(
                matches!(parse(case.clone()), Err(RequestError::Unsupported(_))),
                "{case}"
            );
        }
    }
}
