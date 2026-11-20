// SPDX-License-Identifier: AGPL-3.0-only

//! Responses written in the Anthropic Messages format, for clients that
//! speak it when the provider does not.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::domain::chat::{ChatResponse, ContentPart, FinishReason, StreamEvent, ToolCall, Usage};

fn stop_reason(reason: Option<&FinishReason>) -> Value {
    match reason {
        None => Value::Null,
        Some(FinishReason::Stop) => json!("end_turn"),
        Some(FinishReason::Length) => json!("max_tokens"),
        Some(FinishReason::ToolCalls) => json!("tool_use"),
        Some(FinishReason::ContentFilter) => json!("refusal"),
        Some(FinishReason::Other(other)) => json!(other),
    }
}

fn usage(usage: &Usage) -> Value {
    json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_tokens,
        "cache_creation_input_tokens": usage.cache_write_short_tokens.saturating_add(usage.cache_write_long_tokens),
        "cache_creation": {
            "ephemeral_5m_input_tokens": usage.cache_write_short_tokens,
            "ephemeral_1h_input_tokens": usage.cache_write_long_tokens,
        },
    })
}

// Anthropic clients expect tool input as an object. Arguments that are
// not one (for example cut off by the token limit) become an empty object.
fn tool_input(call: &ToolCall) -> Value {
    match serde_json::from_str::<Value>(&call.arguments) {
        Ok(value @ Value::Object(_)) => value,
        _ => json!({}),
    }
}

/// A non-streamed `message` body.
pub fn render_response(response: &ChatResponse) -> Vec<u8> {
    let content: Vec<Value> = response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text(text) => Some(json!({ "type": "text", "text": text })),
            ContentPart::ToolCall(call) => {
                Some(json!({ "type": "tool_use", "id": call.id, "name": call.name, "input": tool_input(call) }))
            }
            ContentPart::ToolResult { .. } => None,
        })
        .collect();
    json!({
        "id": response.id,
        "type": "message",
        "role": "assistant",
        "model": response.model,
        "content": content,
        "stop_reason": stop_reason(response.finish_reason.as_ref()),
        "stop_sequence": null,
        "usage": usage(&response.usage),
    })
    .to_string()
    .into_bytes()
}

fn event(name: &str, data: Value) -> Vec<u8> {
    format!("event: {name}\ndata: {data}\n\n").into_bytes()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    Text,
    Tool,
}

/// Writes neutral stream events as Anthropic SSE events.
#[derive(Default)]
pub struct StreamEncoder {
    open: Option<(u64, Block)>,
    next_block: u64,
    tool_blocks: HashMap<u32, u64>,
    usage: Usage,
    finish: Option<FinishReason>,
}

impl StreamEncoder {
    /// Creates an encoder for one response.
    pub fn new() -> Self {
        Self::default()
    }

    fn close(&mut self, out: &mut Vec<u8>) {
        if let Some((index, _)) = self.open.take() {
            out.extend(event(
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": index }),
            ));
        }
    }

    fn open(&mut self, kind: Block, block: Value, out: &mut Vec<u8>) -> u64 {
        self.close(out);
        let index = self.next_block;
        self.next_block += 1;
        self.open = Some((index, kind));
        out.extend(event(
            "content_block_start",
            json!({ "type": "content_block_start", "index": index, "content_block": block }),
        ));
        index
    }

    /// Bytes for one event. Usage and the stop reason are held back for
    /// the closing `message_delta`.
    pub fn push(&mut self, item: &StreamEvent) -> Vec<u8> {
        let mut out = Vec::new();
        match item {
            StreamEvent::Start { id, model } => out.extend(event(
                "message_start",
                json!({ "type": "message_start", "message": {
                    "id": id, "type": "message", "role": "assistant", "model": model, "content": [],
                    "stop_reason": null, "stop_sequence": null, "usage": usage(&self.usage),
                } }),
            )),
            StreamEvent::TextDelta(text) => {
                let index = match self.open {
                    Some((index, Block::Text)) => index,
                    _ => self.open(Block::Text, json!({ "type": "text", "text": "" }), &mut out),
                };
                out.extend(event(
                    "content_block_delta",
                    json!({ "type": "content_block_delta", "index": index, "delta": { "type": "text_delta", "text": text } }),
                ));
            }
            StreamEvent::ToolCallStart { index, id, name } => {
                let block = json!({ "type": "tool_use", "id": id, "name": name, "input": {} });
                let block_index = self.open(Block::Tool, block, &mut out);
                self.tool_blocks.insert(*index, block_index);
            }
            StreamEvent::ToolCallDelta { index, arguments } => {
                if let Some(&block_index) = self.tool_blocks.get(index) {
                    out.extend(event(
                        "content_block_delta",
                        json!({ "type": "content_block_delta", "index": block_index,
                                "delta": { "type": "input_json_delta", "partial_json": arguments } }),
                    ));
                }
            }
            StreamEvent::Usage(reported) => self.usage = self.usage.merge(*reported),
            StreamEvent::Finish(reason) => self.finish = Some(reason.clone()),
        }
        out
    }

    /// Closing bytes: the open block's stop, `message_delta` with the stop
    /// reason and usage, and `message_stop`.
    pub fn finish(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        self.close(&mut out);
        out.extend(event(
            "message_delta",
            json!({ "type": "message_delta", "delta": { "stop_reason": stop_reason(self.finish.as_ref()), "stop_sequence": null },
                    "usage": usage(&self.usage) }),
        ));
        out.extend(event("message_stop", json!({ "type": "message_stop" })));
        out
    }

    /// An error after the stream started. Nothing follows it.
    pub fn error(error_type: &str, message: &str) -> Vec<u8> {
        event(
            "error",
            json!({ "type": "error", "error": { "type": error_type, "message": message } }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::ResponseAccumulator;
    use crate::ports::http::fake::ChunkedBody;
    use crate::providers::anthropic::{AnthropicStream, parse_response};
    use crate::providers::openai::{self, OpenAiStream};
    use crate::test_support::collect_body;

    fn fold(events: &[StreamEvent]) -> ChatResponse {
        let mut acc = ResponseAccumulator::new();
        events.iter().for_each(|e| acc.push(e));
        acc.finish()
    }

    #[test]
    fn openai_responses_read_back_the_same_through_the_anthropic_parser() {
        for body in [
            include_str!("../../../fixtures/openai/chat.json"),
            include_str!("../../../fixtures/openai/tool.json"),
            include_str!("../../../fixtures/openai/cache-write.json"),
            include_str!("../../../fixtures/openai/cache-read.json"),
        ] {
            let original = openai::parse_response(200, body.as_bytes()).unwrap();
            let back = parse_response(200, &render_response(&original)).unwrap();
            assert_eq!(back, original);
        }
    }

    #[test]
    fn broken_tool_arguments_become_an_empty_object() {
        let call = ToolCall {
            id: "c".into(),
            name: "t".into(),
            arguments: r#"{"city": "Li"#.into(),
        };
        assert_eq!(tool_input(&call), json!({}));
    }

    #[test]
    fn openai_streams_read_back_the_same_through_the_anthropic_stream() {
        for sse in [
            include_str!("../../../fixtures/openai/chat-stream.sse"),
            include_str!("../../../fixtures/openai/tool-stream.sse"),
        ] {
            let events: Vec<_> =
                collect_body(OpenAiStream::new(ChunkedBody::of(sse.as_bytes(), 7)))
                    .into_iter()
                    .map(Result::unwrap)
                    .collect();
            let mut encoder = StreamEncoder::new();
            let mut bytes: Vec<u8> = events.iter().flat_map(|e| encoder.push(e)).collect();
            bytes.extend(encoder.finish());
            let back: Vec<_> = collect_body(AnthropicStream::new(ChunkedBody::of(&bytes, 5)))
                .into_iter()
                .map(Result::unwrap)
                .collect();
            assert_eq!(fold(&back), fold(&events));
        }
    }

    #[test]
    fn text_after_a_tool_call_opens_a_new_block() {
        let mut encoder = StreamEncoder::new();
        let mut bytes = Vec::new();
        for item in [
            StreamEvent::Start {
                id: "m".into(),
                model: "x".into(),
            },
            StreamEvent::TextDelta("a".into()),
            StreamEvent::ToolCallStart {
                index: 0,
                id: "c".into(),
                name: "t".into(),
            },
            StreamEvent::ToolCallDelta {
                index: 0,
                arguments: "{}".into(),
            },
            StreamEvent::TextDelta("b".into()),
            StreamEvent::Finish(FinishReason::Stop),
        ] {
            bytes.extend(encoder.push(&item));
        }
        bytes.extend(encoder.finish());
        let text = String::from_utf8(bytes).unwrap();
        assert_eq!(text.matches("event: content_block_start").count(), 3);
        assert_eq!(text.matches("event: content_block_stop").count(), 3);
        assert!(text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
        assert!(
            String::from_utf8(StreamEncoder::error("overloaded_error", "busy"))
                .unwrap()
                .starts_with("event: error\n")
        );
    }
}
