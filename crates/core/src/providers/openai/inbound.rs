// SPDX-License-Identifier: AGPL-3.0-only

//! OpenAI-format client requests read into the neutral form, for sending
//! to a provider with another format. Same-format requests are forwarded
//! as is and never pass through here.

use serde_json::{Value, json};

use crate::domain::chat::{
    ChatRequest, ContentPart, Message, Role, ToolCall, ToolChoice, ToolDefinition,
};
use crate::providers::RequestError;

fn invalid(what: &str) -> RequestError {
    RequestError::Invalid(what.to_string())
}

/// Reads a Chat Completions request body.
pub fn parse_request(body: &[u8]) -> Result<ChatRequest, RequestError> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|e| RequestError::Invalid(format!("body is not JSON: {e}")))?;
    let model = value["model"]
        .as_str()
        .ok_or_else(|| invalid("model must be a string"))?;
    if value["n"].as_u64().is_some_and(|n| n > 1) {
        return Err(RequestError::Unsupported("n greater than 1".into()));
    }
    let mut system = Vec::new();
    let mut messages: Vec<Message> = Vec::new();
    for m in value["messages"]
        .as_array()
        .ok_or_else(|| invalid("messages must be an array"))?
    {
        let role = m["role"]
            .as_str()
            .ok_or_else(|| invalid("every message needs a role"))?;
        let (role, parts) = match role {
            "system" | "developer" => {
                system.push(text_of(&m["content"])?);
                continue;
            }
            "user" => (Role::User, vec![ContentPart::Text(text_of(&m["content"])?)]),
            "assistant" => (Role::Assistant, assistant_parts(m)?),
            "tool" => {
                let call_id = m["tool_call_id"]
                    .as_str()
                    .ok_or_else(|| invalid("tool messages need tool_call_id"))?;
                let content = text_of(&m["content"])?;
                (
                    Role::User,
                    vec![ContentPart::ToolResult {
                        call_id: call_id.to_string(),
                        content,
                        is_error: false,
                    }],
                )
            }
            other => {
                return Err(RequestError::Invalid(format!(
                    "unknown message role {other:?}"
                )));
            }
        };
        // Anthropic expects tool results and the user text after them in
        // one turn, so adjacent turns of the same role are merged.
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
        system: (!system.is_empty()).then(|| system.join("\n\n")),
        messages,
        tools: tools(&value["tools"])?,
        tool_choice: tool_choice(&value["tool_choice"])?,
        max_tokens: optional_u32(&value, "max_completion_tokens")?
            .or(optional_u32(&value, "max_tokens")?),
        temperature: optional_f64(&value, "temperature")?,
        top_p: optional_f64(&value, "top_p")?,
        stop: match &value["stop"] {
            Value::Null => Vec::new(),
            Value::String(s) => vec![s.clone()],
            Value::Array(items) => items
                .iter()
                .map(|s| {
                    s.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| invalid("stop must hold strings"))
                })
                .collect::<Result<_, _>>()?,
            _ => return Err(invalid("stop must be a string or an array")),
        },
        stream: value["stream"].as_bool().unwrap_or(false),
    })
}

#[derive(serde::Deserialize)]
struct StreamOptionsOnly {
    stream_options: Option<StreamOptions>,
}

#[derive(serde::Deserialize)]
struct StreamOptions {
    include_usage: Option<bool>,
}

/// Whether a request asks for the closing usage chunk of a stream, which
/// the neutral form does not carry. Other fields are scanned, not kept.
pub fn include_usage(body: &[u8]) -> bool {
    serde_json::from_slice::<StreamOptionsOnly>(body)
        .ok()
        .and_then(|r| r.stream_options)
        .and_then(|o| o.include_usage)
        .unwrap_or(false)
}

// Text content: a string, or an array of parts that are all text.
fn text_of(content: &Value) -> Result<String, RequestError> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => parts
            .iter()
            .map(|part| match part["type"].as_str() {
                Some("text") => part["text"]
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| invalid("text parts need text")),
                Some(other) => Err(RequestError::Unsupported(format!("{other} content parts"))),
                None => Err(invalid("content parts need a type")),
            })
            .collect(),
        _ => Err(invalid("content must be a string or an array of parts")),
    }
}

fn assistant_parts(message: &Value) -> Result<Vec<ContentPart>, RequestError> {
    let mut parts = Vec::new();
    let text = text_of(&message["content"])?;
    if !text.is_empty() {
        parts.push(ContentPart::Text(text));
    }
    for call in message["tool_calls"].as_array().into_iter().flatten() {
        let function = &call["function"];
        let (Some(id), Some(name)) = (call["id"].as_str(), function["name"].as_str()) else {
            return Err(invalid("tool calls need an id and a function name"));
        };
        parts.push(ContentPart::ToolCall(ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: function["arguments"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
        }));
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
            if tool["type"] != "function" {
                return Err(RequestError::Unsupported(
                    "tools other than functions".into(),
                ));
            }
            let function = &tool["function"];
            let name = function["name"]
                .as_str()
                .ok_or_else(|| invalid("functions need a name"))?;
            Ok(ToolDefinition {
                name: name.to_string(),
                description: function["description"].as_str().map(str::to_string),
                parameters: match &function["parameters"] {
                    Value::Null => json!({ "type": "object", "properties": {} }),
                    schema => schema.clone(),
                },
            })
        })
        .collect()
}

fn tool_choice(value: &Value) -> Result<Option<ToolChoice>, RequestError> {
    Ok(match value {
        Value::Null => None,
        Value::String(s) => Some(match s.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            other => {
                return Err(RequestError::Invalid(format!(
                    "unknown tool_choice {other:?}"
                )));
            }
        }),
        Value::Object(_) => {
            let name = value["function"]["name"]
                .as_str()
                .ok_or_else(|| invalid("tool_choice needs a function name"))?;
            Some(ToolChoice::Tool(name.to_string()))
        }
        _ => return Err(invalid("tool_choice must be a string or an object")),
    })
}

fn optional_u32(value: &Value, key: &str) -> Result<Option<u32>, RequestError> {
    match &value[key] {
        Value::Null => Ok(None),
        v => v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| RequestError::Invalid(format!("{key} must be a non-negative integer"))),
    }
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
    use super::*;

    fn parse(value: Value) -> Result<ChatRequest, RequestError> {
        parse_request(value.to_string().as_bytes())
    }

    #[test]
    fn reads_a_full_conversation() {
        let request = parse(json!({
            "model": "gpt-6-luna",
            "messages": [
                { "role": "developer", "content": "Be brief." },
                { "role": "system", "content": [{ "type": "text", "text": "Answer in English." }] },
                { "role": "user", "content": "Weather in Lima?" },
                { "role": "assistant", "content": null, "tool_calls": [
                    { "id": "call_1", "type": "function", "function": { "name": "get_weather", "arguments": "{\"city\":\"Lima\"}" } },
                ] },
                { "role": "tool", "tool_call_id": "call_1", "content": "18 C" },
                { "role": "user", "content": [{ "type": "text", "text": "And " }, { "type": "text", "text": "tomorrow?" }] },
            ],
            "tools": [{ "type": "function", "function": { "name": "get_weather", "description": "Weather.", "parameters": { "type": "object" } } }],
            "tool_choice": { "type": "function", "function": { "name": "get_weather" } },
            "max_completion_tokens": 300,
            "temperature": 0.2,
            "stop": "END",
            "stream": true,
            "seed": 7,
        }))
        .unwrap();
        assert_eq!(request.model, "gpt-6-luna");
        assert_eq!(
            request.system.as_deref(),
            Some("Be brief.\n\nAnswer in English.")
        );
        assert_eq!(
            request.messages,
            [
                Message {
                    role: Role::User,
                    content: vec![ContentPart::Text("Weather in Lima?".into())]
                },
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
                        ContentPart::ToolResult {
                            call_id: "call_1".into(),
                            content: "18 C".into(),
                            is_error: false
                        },
                        ContentPart::Text("And tomorrow?".into()),
                    ],
                },
            ]
        );
        assert_eq!(request.tools[0].name, "get_weather");
        assert_eq!(
            request.tool_choice,
            Some(ToolChoice::Tool("get_weather".into()))
        );
        assert_eq!(
            (
                request.max_tokens,
                request.temperature,
                request.stop.clone(),
                request.stream
            ),
            (Some(300), Some(0.2), vec!["END".to_string()], true)
        );
    }

    #[test]
    fn minimal_request_and_legacy_fields() {
        let request = parse(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 50,
            "tools": [{ "type": "function", "function": { "name": "now" } }],
            "tool_choice": "required",
        }))
        .unwrap();
        assert_eq!(
            (request.system, request.max_tokens, request.stream),
            (None, Some(50), false)
        );
        assert_eq!(
            request.tools[0].parameters,
            json!({ "type": "object", "properties": {} })
        );
        assert_eq!(request.tool_choice, Some(ToolChoice::Required));
    }

    #[test]
    fn rejects_invalid_requests() {
        let cases = [
            json!({ "messages": [] }),
            json!({ "model": "m" }),
            json!({ "model": "m", "messages": [{ "content": "hi" }] }),
            json!({ "model": "m", "messages": [{ "role": "robot", "content": "hi" }] }),
            json!({ "model": "m", "messages": [{ "role": "tool", "content": "x" }] }),
            json!({ "model": "m", "messages": [], "max_tokens": -1 }),
            json!({ "model": "m", "messages": [], "temperature": "warm" }),
            json!({ "model": "m", "messages": [], "tool_choice": "sometimes" }),
            json!({ "model": "m", "messages": [], "stop": [1] }),
        ];
        for case in cases {
            assert!(
                matches!(parse(case.clone()), Err(RequestError::Invalid(_))),
                "{case}"
            );
        }
        assert!(matches!(
            parse_request(b"{not json"),
            Err(RequestError::Invalid(_))
        ));
    }

    #[test]
    fn reports_what_cannot_be_translated_yet() {
        let image =
            json!({ "type": "image_url", "image_url": { "url": "https://example.test/a.png" } });
        let cases = [
            json!({ "model": "m", "messages": [{ "role": "user", "content": [image] }] }),
            json!({ "model": "m", "messages": [], "n": 2 }),
            json!({ "model": "m", "messages": [], "tools": [{ "type": "custom", "custom": { "name": "x" } }] }),
        ];
        for case in cases {
            assert!(
                matches!(parse(case.clone()), Err(RequestError::Unsupported(_))),
                "{case}"
            );
        }
    }
}
