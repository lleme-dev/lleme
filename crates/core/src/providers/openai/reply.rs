// SPDX-License-Identifier: AGPL-3.0-only

//! Responses written in the OpenAI Chat Completions format, for clients
//! that speak it when the provider does not.

use serde_json::{Value, json};

use crate::domain::chat::{ChatResponse, ContentPart, FinishReason, StreamEvent, Usage};

fn finish_reason(reason: Option<&FinishReason>) -> Value {
    match reason {
        None => Value::Null,
        Some(FinishReason::Stop) => json!("stop"),
        Some(FinishReason::Length) => json!("length"),
        Some(FinishReason::ToolCalls) => json!("tool_calls"),
        Some(FinishReason::ContentFilter) => json!("content_filter"),
        // OpenAI clients know only the values above.
        Some(FinishReason::Other(_)) => json!("stop"),
    }
}

// OpenAI counts cached and cache-written tokens inside prompt_tokens.
fn usage(usage: &Usage) -> Value {
    let written = usage
        .cache_write_short_tokens
        .saturating_add(usage.cache_write_long_tokens);
    let prompt = usage.total_input_tokens();
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": usage.output_tokens,
        "total_tokens": prompt.saturating_add(usage.output_tokens),
        "prompt_tokens_details": { "cached_tokens": usage.cache_read_tokens, "cache_write_tokens": written },
    })
}

/// A non-streamed `chat.completion` body. `created` is Unix seconds.
pub fn render_response(response: &ChatResponse, created: u64) -> Vec<u8> {
    let text: String = response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    let calls: Vec<Value> = response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(json!({
                "id": call.id,
                "type": "function",
                "function": { "name": call.name, "arguments": call.arguments },
            })),
            _ => None,
        })
        .collect();
    let mut message = json!({ "role": "assistant", "content": if text.is_empty() { Value::Null } else { json!(text) } });
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls);
    }
    json!({
        "id": response.id,
        "object": "chat.completion",
        "created": created,
        "model": response.model,
        "choices": [{ "index": 0, "message": message, "finish_reason": finish_reason(response.finish_reason.as_ref()) }],
        "usage": usage(&response.usage),
    })
    .to_string()
    .into_bytes()
}

/// Writes neutral stream events as `chat.completion.chunk` SSE events.
pub struct StreamEncoder {
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
    usage: Usage,
}

impl StreamEncoder {
    /// `include_usage` mirrors the client's `stream_options.include_usage`.
    pub fn new(created: u64, include_usage: bool) -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            created,
            include_usage,
            usage: Usage::default(),
        }
    }

    fn chunk(&self, delta: Value, finish: Value) -> Vec<u8> {
        let chunk = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
        });
        format!("data: {chunk}\n\n").into_bytes()
    }

    /// Bytes for one event; usage is held back for the final chunk.
    pub fn push(&mut self, event: &StreamEvent) -> Vec<u8> {
        match event {
            StreamEvent::Start { id, model } => {
                self.id.clone_from(id);
                self.model.clone_from(model);
                self.chunk(json!({ "role": "assistant", "content": "" }), Value::Null)
            }
            StreamEvent::TextDelta(text) => self.chunk(json!({ "content": text }), Value::Null),
            StreamEvent::ToolCallStart { index, id, name } => self.chunk(
                json!({ "tool_calls": [{ "index": index, "id": id, "type": "function", "function": { "name": name, "arguments": "" } }] }),
                Value::Null,
            ),
            StreamEvent::ToolCallDelta { index, arguments } => {
                self.chunk(json!({ "tool_calls": [{ "index": index, "function": { "arguments": arguments } }] }), Value::Null)
            }
            StreamEvent::Usage(usage) => {
                self.usage = self.usage.merge(*usage);
                Vec::new()
            }
            StreamEvent::Finish(reason) => self.chunk(json!({}), finish_reason(Some(reason))),
        }
    }

    /// Closing bytes: the usage chunk if the client asked for it, then `[DONE]`.
    pub fn finish(&self) -> Vec<u8> {
        let mut out = Vec::new();
        if self.include_usage {
            let chunk = json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [],
                "usage": usage(&self.usage),
            });
            out.extend(format!("data: {chunk}\n\n").into_bytes());
        }
        out.extend(b"data: [DONE]\n\n");
        out
    }

    /// An error after the stream started. No `[DONE]` follows.
    pub fn error(error_type: &str, message: &str) -> Vec<u8> {
        format!(
            "data: {}\n\n",
            json!({ "error": { "message": message, "type": error_type } })
        )
        .into_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::ResponseAccumulator;
    use crate::ports::http::fake::ChunkedBody;
    use crate::providers::anthropic;
    use crate::providers::openai::{OpenAiStream, parse_response};
    use crate::providers::sse::SseParser;
    use crate::test_support::collect_body;

    fn anthropic_fixture(name: &str) -> ChatResponse {
        let body = match name {
            "chat" => include_str!("../../../fixtures/anthropic/chat.json"),
            "tool" => include_str!("../../../fixtures/anthropic/tool.json"),
            _ => include_str!("../../../fixtures/anthropic/cache-write.json"),
        };
        anthropic::parse_response(200, body.as_bytes()).unwrap()
    }

    #[test]
    fn anthropic_responses_read_back_the_same_through_the_openai_parser() {
        for name in ["chat", "tool", "cache"] {
            let original = anthropic_fixture(name);
            let rendered = render_response(&original, 1_700_000_000);
            let back = parse_response(200, &rendered).unwrap();
            assert_eq!(back.content, original.content, "{name}");
            assert_eq!(back.finish_reason, original.finish_reason, "{name}");
            // OpenAI reports no write lifetime, so long writes come back short.
            let mut expected = original.usage;
            expected.cache_write_short_tokens += expected.cache_write_long_tokens;
            expected.cache_write_long_tokens = 0;
            assert_eq!(back.usage, expected, "{name}");
        }
        let body: Value =
            serde_json::from_slice(&render_response(&anthropic_fixture("chat"), 7)).unwrap();
        assert_eq!(
            (body["created"].clone(), body["object"].clone()),
            (json!(7), json!("chat.completion"))
        );
    }

    fn anthropic_events(sse: &str) -> Vec<StreamEvent> {
        let body = ChunkedBody::of(sse.as_bytes(), 7);
        collect_body(anthropic::AnthropicStream::new(body))
            .into_iter()
            .map(Result::unwrap)
            .collect()
    }

    fn fold(events: &[StreamEvent]) -> ChatResponse {
        let mut acc = ResponseAccumulator::new();
        events.iter().for_each(|e| acc.push(e));
        acc.finish()
    }

    fn encode(events: &[StreamEvent], include_usage: bool) -> Vec<u8> {
        let mut encoder = StreamEncoder::new(1_700_000_000, include_usage);
        let mut out: Vec<u8> = events.iter().flat_map(|e| encoder.push(e)).collect();
        out.extend(encoder.finish());
        out
    }

    #[test]
    fn anthropic_streams_read_back_the_same_through_the_openai_stream() {
        for sse in [
            include_str!("../../../fixtures/anthropic/chat-stream.sse"),
            include_str!("../../../fixtures/anthropic/tool-stream.sse"),
        ] {
            let events = anthropic_events(sse);
            let bytes = encode(&events, true);
            let back: Vec<_> = collect_body(OpenAiStream::new(ChunkedBody::of(&bytes, 5)))
                .into_iter()
                .map(Result::unwrap)
                .collect();
            let (mut original, back) = (fold(&events), fold(&back));
            original.usage.cache_write_short_tokens += original.usage.cache_write_long_tokens;
            original.usage.cache_write_long_tokens = 0;
            assert_eq!(back, original);
        }
    }

    #[test]
    fn usage_chunk_only_when_asked() {
        let events = [StreamEvent::Usage(Usage {
            input_tokens: 3,
            ..Usage::default()
        })];
        let without = String::from_utf8(encode(&events, false)).unwrap();
        assert_eq!(without, "data: [DONE]\n\n");
        let with = String::from_utf8(encode(&events, true)).unwrap();
        let mut parser = SseParser::new();
        let parsed = parser.push(with.as_bytes()).unwrap();
        assert_eq!(parsed.len(), 2);
        assert!(parsed[0].data.contains("\"prompt_tokens\":3"));
        assert!(
            String::from_utf8(StreamEncoder::error("overloaded_error", "busy"))
                .unwrap()
                .contains("\"error\"")
        );
    }
}
