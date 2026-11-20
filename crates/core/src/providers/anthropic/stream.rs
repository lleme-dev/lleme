// SPDX-License-Identifier: AGPL-3.0-only

//! Streamed Anthropic responses: SSE events translated to `StreamEvent`s
//! as the bytes arrive, so the client gets each event without waiting for
//! the end of the response.

use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use serde_json::Value;

use super::{AnthropicConfig, build_request, finish_reason, status_error, usage};
use crate::domain::chat::{ChatRequest, StreamEvent};
use crate::ports::{HttpClient, HttpError, read_to_end};
use crate::providers::ProviderError;
use crate::providers::sse::{SseEvent, SseParser};

/// Sends a streamed request. An error status before the stream starts is
/// returned here; errors after it come through the stream.
pub async fn stream<H: HttpClient>(
    client: &H,
    config: &AnthropicConfig,
    request: &ChatRequest,
    default_max_tokens: u32,
) -> Result<AnthropicStream<H::Body>, ProviderError> {
    let http = build_request(config, request, default_max_tokens, true)?;
    let response = client.send(http).await?;
    if !(200..300).contains(&response.status) {
        let body = read_to_end(response.body).await?;
        return Err(status_error(response.status, &body));
    }
    Ok(AnthropicStream::new(response.body))
}

/// The events of one streamed response. Ends after `message_stop`, or
/// after the first error.
pub struct AnthropicStream<B> {
    body: B,
    parser: SseParser,
    ready: VecDeque<Result<StreamEvent, ProviderError>>,
    // Anthropic numbers content blocks; the neutral form numbers tool calls.
    tool_index: HashMap<u64, u32>,
    finished: bool,
}

impl<B> AnthropicStream<B>
where
    B: Stream<Item = Result<Vec<u8>, HttpError>> + Unpin,
{
    pub(crate) fn new(body: B) -> Self {
        Self {
            body,
            parser: SseParser::new(),
            ready: VecDeque::new(),
            tool_index: HashMap::new(),
            finished: false,
        }
    }

    fn fail(&mut self, error: ProviderError) {
        self.ready.push_back(Err(error));
        self.finished = true;
    }

    fn translate(&mut self, event: SseEvent) {
        let Ok(value) = serde_json::from_str::<Value>(&event.data) else {
            return self.fail(ProviderError::Malformed(
                "anthropic stream event is not JSON".into(),
            ));
        };
        let text = |v: &Value| v.as_str().unwrap_or_default().to_string();
        let index = value["index"].as_u64();
        match value["type"].as_str() {
            Some("message_start") => {
                let message = &value["message"];
                self.ready.push_back(Ok(StreamEvent::Start {
                    id: text(&message["id"]),
                    model: text(&message["model"]),
                }));
                self.ready
                    .push_back(Ok(StreamEvent::Usage(usage(&message["usage"]))));
            }
            Some("content_block_start") => {
                let block = &value["content_block"];
                match block["type"].as_str() {
                    Some("text") if !text(&block["text"]).is_empty() => {
                        self.ready
                            .push_back(Ok(StreamEvent::TextDelta(text(&block["text"]))));
                    }
                    Some("tool_use") => {
                        let position = u32::try_from(self.tool_index.len()).unwrap_or(u32::MAX);
                        self.tool_index.insert(index.unwrap_or(u64::MAX), position);
                        self.ready.push_back(Ok(StreamEvent::ToolCallStart {
                            index: position,
                            id: text(&block["id"]),
                            name: text(&block["name"]),
                        }));
                    }
                    _ => {}
                }
            }
            Some("content_block_delta") => {
                let delta = &value["delta"];
                match delta["type"].as_str() {
                    Some("text_delta") => self
                        .ready
                        .push_back(Ok(StreamEvent::TextDelta(text(&delta["text"])))),
                    Some("input_json_delta") => {
                        let fragment = text(&delta["partial_json"]);
                        let position = index.and_then(|i| self.tool_index.get(&i)).copied();
                        if let (Some(index), false) = (position, fragment.is_empty()) {
                            self.ready.push_back(Ok(StreamEvent::ToolCallDelta {
                                index,
                                arguments: fragment,
                            }));
                        }
                    }
                    // Thinking and signature deltas have no neutral form yet.
                    _ => {}
                }
            }
            Some("message_delta") => {
                self.ready
                    .push_back(Ok(StreamEvent::Usage(usage(&value["usage"]))));
                if let Some(reason) = value["delta"]["stop_reason"].as_str() {
                    self.ready
                        .push_back(Ok(StreamEvent::Finish(finish_reason(reason))));
                }
            }
            Some("message_stop") => self.finished = true,
            Some("error") => {
                let error = &value["error"];
                self.fail(ProviderError::Interrupted {
                    error_type: error["type"].as_str().map(str::to_string),
                    message: text(&error["message"]),
                });
            }
            // ping, content_block_stop and event types added later.
            _ => {}
        }
    }
}

impl<B> Stream for AnthropicStream<B>
where
    B: Stream<Item = Result<Vec<u8>, HttpError>> + Unpin,
{
    type Item = Result<StreamEvent, ProviderError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            if let Some(item) = this.ready.pop_front() {
                return Poll::Ready(Some(item));
            }
            if this.finished {
                return Poll::Ready(None);
            }
            match Pin::new(&mut this.body).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(chunk))) => match this.parser.push(&chunk) {
                    Ok(events) => {
                        for event in events {
                            if this.finished {
                                break;
                            }
                            this.translate(event);
                        }
                    }
                    Err(error) => this.fail(error),
                },
                Poll::Ready(Some(Err(error))) => this.fail(ProviderError::Http(error)),
                Poll::Ready(None) => this.fail(ProviderError::Malformed(
                    "anthropic stream ended before message_stop".into(),
                )),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::domain::chat::{ContentPart, FinishReason, ResponseAccumulator, ToolCall, Usage};
    use crate::ports::http::fake::FakeHttpClient;
    use crate::test_support::{block_on, collect_body};

    const CHAT: &str = include_str!("../../../fixtures/anthropic/chat-stream.sse");
    const TOOL: &str = include_str!("../../../fixtures/anthropic/tool-stream.sse");

    fn config() -> AnthropicConfig {
        AnthropicConfig {
            base_url: "https://anthropic.test".into(),
            api_key: "k".into(),
        }
    }

    fn request() -> ChatRequest {
        ChatRequest {
            model: "claude-haiku-5-5".into(),
            system: None,
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            stream: true,
        }
    }

    fn events(
        body: &str,
        status: u16,
        chunk: usize,
    ) -> Result<Vec<Result<StreamEvent, ProviderError>>, ProviderError> {
        let client = FakeHttpClient::new(status, body, chunk);
        let sent = block_on(stream(&client, &config(), &request(), 64));
        let body: Value = serde_json::from_slice(&client.requests()[0].body).unwrap();
        assert_eq!(body["stream"], true);
        sent.map(collect_body)
    }

    fn fold(events: &[Result<StreamEvent, ProviderError>]) -> crate::domain::chat::ChatResponse {
        let mut acc = ResponseAccumulator::new();
        for event in events {
            acc.push(event.as_ref().unwrap());
        }
        acc.finish()
    }

    #[test]
    fn streamed_chat_in_small_chunks() {
        let events = events(CHAT, 200, 7).unwrap();
        let usage = |output| {
            StreamEvent::Usage(Usage {
                input_tokens: 34,
                output_tokens: output,
                ..Usage::default()
            })
        };
        assert_eq!(
            events,
            [
                Ok(StreamEvent::Start {
                    id: "msg_fixture01".into(),
                    model: "claude-haiku-5-5".into()
                }),
                Ok(usage(8)),
                Ok(StreamEvent::TextDelta("Hello from the fixture".into())),
                Ok(StreamEvent::TextDelta(".".into())),
                Ok(usage(11)),
                Ok(StreamEvent::Finish(FinishReason::Stop)),
            ]
        );
        let response = fold(&events);
        assert_eq!(
            response.content,
            [ContentPart::Text("Hello from the fixture.".into())]
        );
        assert_eq!(
            response.usage,
            Usage {
                input_tokens: 34,
                output_tokens: 11,
                ..Usage::default()
            }
        );
    }

    #[test]
    fn streamed_tool_call() {
        let events = events(TOOL, 200, 7).unwrap();
        assert!(events.contains(&Ok(StreamEvent::ToolCallStart {
            index: 0,
            id: "toolu_fixture02".into(),
            name: "get_weather".into(),
        })));
        let response = fold(&events);
        assert_eq!(
            response.content,
            [ContentPart::ToolCall(ToolCall {
                id: "toolu_fixture02".into(),
                name: "get_weather".into(),
                arguments: r#"{"city": "Lima"}"#.into(),
            })]
        );
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(response.usage.output_tokens, 50);
    }

    // A body that counts how many chunks have been read from it.
    struct CountingBody {
        chunks: VecDeque<Vec<u8>>,
        read: Arc<AtomicUsize>,
    }

    impl Stream for CountingBody {
        type Item = Result<Vec<u8>, HttpError>;

        fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let next = self.chunks.pop_front();
            if next.is_some() {
                self.read.fetch_add(1, Ordering::SeqCst);
            }
            Poll::Ready(next.map(Ok))
        }
    }

    #[test]
    fn first_event_comes_out_before_the_last_chunk_is_read() {
        let read = Arc::new(AtomicUsize::new(0));
        let chunks: VecDeque<Vec<u8>> = CHAT.as_bytes().chunks(7).map(<[u8]>::to_vec).collect();
        let total = chunks.len();
        let mut events = AnthropicStream::new(CountingBody {
            chunks,
            read: Arc::clone(&read),
        });
        let mut cx = Context::from_waker(std::task::Waker::noop());
        let first = Pin::new(&mut events).poll_next(&mut cx);
        assert!(matches!(
            first,
            Poll::Ready(Some(Ok(StreamEvent::Start { .. })))
        ));
        assert!(
            read.load(Ordering::SeqCst) < total / 2,
            "read {} of {total} chunks",
            read.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn error_status_before_the_stream() {
        let body = include_str!("../../../fixtures/anthropic/error-404.json");
        assert!(matches!(
            events(body, 404, 7),
            Err(ProviderError::Status { status: 404, .. })
        ));
    }

    #[test]
    fn error_event_mid_stream() {
        let start = &CHAT[..CHAT.find("event: content_block_start").unwrap()];
        let body = format!(
            "{start}event: error\ndata: {{\"type\":\"error\",\"error\":{{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}}}\n\n"
        );
        let events = events(&body, 200, 7).unwrap();
        assert_eq!(
            events.last(),
            Some(&Err(ProviderError::Interrupted {
                error_type: Some("overloaded_error".into()),
                message: "Overloaded".into()
            }))
        );
        assert!(matches!(events[0], Ok(StreamEvent::Start { .. })));
    }

    #[test]
    fn cut_off_and_malformed_streams() {
        let cut = &CHAT[..CHAT.find("event: message_delta").unwrap()];
        let events_cut = events(cut, 200, 7).unwrap();
        assert!(matches!(
            events_cut.last(),
            Some(Err(ProviderError::Malformed(_)))
        ));
        let garbage = events("event: message_start\ndata: {not json\n\n", 200, 7).unwrap();
        assert!(matches!(
            garbage.as_slice(),
            [Err(ProviderError::Malformed(_))]
        ));
    }

    #[test]
    fn body_that_breaks_off() {
        let client = FakeHttpClient::new(200, CHAT, 16).failing_after(3);
        let events = collect_body(block_on(stream(&client, &config(), &request(), 64)).unwrap());
        assert_eq!(
            events.last(),
            Some(&Err(ProviderError::Http(HttpError::Body(
                "connection reset".into()
            ))))
        );
    }
}
