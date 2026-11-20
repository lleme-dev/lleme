// SPDX-License-Identifier: AGPL-3.0-only

//! Streamed OpenAI responses: `chat.completion.chunk` events translated to
//! `StreamEvent`s as the bytes arrive.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use serde_json::Value;

use super::{ModelLimits, OpenAiConfig, build_request, finish_reason, status_error, usage};
use crate::domain::chat::{ChatRequest, StreamEvent};
use crate::ports::{HttpClient, HttpError, read_to_end};
use crate::providers::ProviderError;
use crate::providers::sse::{SseEvent, SseParser};

/// Sends a streamed request. An error status before the stream starts is
/// returned here; errors after it come through the stream.
pub async fn stream<H: HttpClient>(
    client: &H,
    config: &OpenAiConfig,
    request: &ChatRequest,
    limits: ModelLimits,
) -> Result<OpenAiStream<H::Body>, ProviderError> {
    let http = build_request(config, request, limits, true)?;
    let response = client.send(http).await?;
    if !(200..300).contains(&response.status) {
        let body = read_to_end(response.body).await?;
        return Err(status_error(response.status, &body));
    }
    Ok(OpenAiStream::new(response.body))
}

/// The events of one streamed response. Ends after `[DONE]`, or after the
/// first error.
pub struct OpenAiStream<B> {
    body: B,
    parser: SseParser,
    ready: VecDeque<Result<StreamEvent, ProviderError>>,
    started: bool,
    finished: bool,
}

impl<B> OpenAiStream<B>
where
    B: Stream<Item = Result<Vec<u8>, HttpError>> + Unpin,
{
    pub(crate) fn new(body: B) -> Self {
        Self {
            body,
            parser: SseParser::new(),
            ready: VecDeque::new(),
            started: false,
            finished: false,
        }
    }

    fn fail(&mut self, error: ProviderError) {
        self.ready.push_back(Err(error));
        self.finished = true;
    }

    fn translate(&mut self, event: SseEvent) {
        if event.data == "[DONE]" {
            self.finished = true;
            return;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(&event.data) else {
            return self.fail(ProviderError::Malformed(
                "openai stream chunk is not JSON".into(),
            ));
        };
        let text = |v: &Value| v.as_str().unwrap_or_default().to_string();
        if chunk["error"].is_object() {
            let error = &chunk["error"];
            return self.fail(ProviderError::Interrupted {
                error_type: error["code"]
                    .as_str()
                    .or_else(|| error["type"].as_str())
                    .map(str::to_string),
                message: text(&error["message"]),
            });
        }
        if !self.started && chunk["id"].is_string() {
            self.started = true;
            self.ready.push_back(Ok(StreamEvent::Start {
                id: text(&chunk["id"]),
                model: text(&chunk["model"]),
            }));
        }
        if let Some(choice) = chunk["choices"].get(0) {
            let delta = &choice["delta"];
            if let Some(content) = delta["content"].as_str().filter(|c| !c.is_empty()) {
                self.ready
                    .push_back(Ok(StreamEvent::TextDelta(content.to_string())));
            }
            for call in delta["tool_calls"].as_array().into_iter().flatten() {
                let Some(index) = call["index"].as_u64().and_then(|i| u32::try_from(i).ok()) else {
                    return self.fail(ProviderError::Malformed(
                        "openai tool call delta has no index".into(),
                    ));
                };
                if let Some(id) = call["id"].as_str() {
                    let name = text(&call["function"]["name"]);
                    self.ready.push_back(Ok(StreamEvent::ToolCallStart {
                        index,
                        id: id.to_string(),
                        name,
                    }));
                }
                if let Some(arguments) = call["function"]["arguments"]
                    .as_str()
                    .filter(|a| !a.is_empty())
                {
                    self.ready.push_back(Ok(StreamEvent::ToolCallDelta {
                        index,
                        arguments: arguments.to_string(),
                    }));
                }
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                self.ready
                    .push_back(Ok(StreamEvent::Finish(finish_reason(reason))));
            }
        }
        if chunk["usage"].is_object() {
            self.ready
                .push_back(Ok(StreamEvent::Usage(usage(&chunk["usage"]))));
        }
    }
}

impl<B> Stream for OpenAiStream<B>
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
                Poll::Ready(Some(Ok(bytes))) => match this.parser.push(&bytes) {
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
                    "openai stream ended before [DONE]".into(),
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
    use crate::domain::chat::{
        ChatResponse, ContentPart, FinishReason, ResponseAccumulator, ToolCall, Usage,
    };
    use crate::ports::http::fake::FakeHttpClient;
    use crate::test_support::{block_on, collect_body};

    const CHAT: &str = include_str!("../../../fixtures/openai/chat-stream.sse");
    const TOOL: &str = include_str!("../../../fixtures/openai/tool-stream.sse");

    fn config() -> OpenAiConfig {
        OpenAiConfig {
            base_url: "https://openai.test".into(),
            api_key: "k".into(),
        }
    }

    fn request() -> ChatRequest {
        ChatRequest {
            model: "gpt-6-luna".into(),
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

    type Events = Vec<Result<StreamEvent, ProviderError>>;

    fn events(body: &str, status: u16, chunk: usize) -> Result<Events, ProviderError> {
        let client = FakeHttpClient::new(status, body, chunk);
        let sent = block_on(stream(
            &client,
            &config(),
            &request(),
            ModelLimits::default(),
        ));
        let body: Value = serde_json::from_slice(&client.requests()[0].body).unwrap();
        assert_eq!(
            (
                body["stream"].clone(),
                body["stream_options"]["include_usage"].clone()
            ),
            (true.into(), true.into())
        );
        sent.map(collect_body)
    }

    fn fold(events: &Events) -> ChatResponse {
        let mut acc = ResponseAccumulator::new();
        for event in events {
            acc.push(event.as_ref().unwrap());
        }
        acc.finish()
    }

    #[test]
    fn streamed_chat_in_small_chunks() {
        let events = events(CHAT, 200, 7).unwrap();
        let deltas: Vec<_> = ["Hello", " from", " the", " fixture", "."]
            .into_iter()
            .map(|t| Ok(StreamEvent::TextDelta(t.into())))
            .collect();
        let mut expected = vec![Ok(StreamEvent::Start {
            id: "chatcmpl-fixture01".into(),
            model: "gpt-6-luna".into(),
        })];
        expected.extend(deltas);
        expected.push(Ok(StreamEvent::Finish(FinishReason::Stop)));
        expected.push(Ok(StreamEvent::Usage(Usage {
            input_tokens: 20,
            output_tokens: 8,
            ..Usage::default()
        })));
        assert_eq!(events, expected);
        assert_eq!(
            fold(&events).content,
            [ContentPart::Text("Hello from the fixture.".into())]
        );
    }

    #[test]
    fn streamed_tool_call() {
        let events = events(TOOL, 200, 7).unwrap();
        let response = fold(&events);
        assert_eq!(
            response.content,
            [ContentPart::ToolCall(ToolCall {
                id: "call_fixture02".into(),
                name: "get_weather".into(),
                arguments: r#"{"city":"Lima"}"#.into(),
            })]
        );
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(
            response.usage,
            Usage {
                input_tokens: 135,
                output_tokens: 18,
                ..Usage::default()
            }
        );
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
        let mut events = OpenAiStream::new(CountingBody {
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
            read.load(Ordering::SeqCst) < total / 4,
            "read {} of {total} chunks",
            read.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn error_status_before_the_stream() {
        let body = include_str!("../../../fixtures/openai/error-404.json");
        assert!(matches!(
            events(body, 404, 7),
            Err(ProviderError::Status { status: 404, .. })
        ));
    }

    #[test]
    fn error_chunk_mid_stream() {
        let first = &CHAT[..CHAT.find("\n\n").unwrap() + 2];
        let body = format!(
            "{first}data: {{\"error\":{{\"message\":\"Overloaded\",\"type\":\"server_error\",\"code\":null}}}}\n\n"
        );
        let events = events(&body, 200, 7).unwrap();
        assert!(matches!(events[0], Ok(StreamEvent::Start { .. })));
        assert_eq!(
            events.last(),
            Some(&Err(ProviderError::Interrupted {
                error_type: Some("server_error".into()),
                message: "Overloaded".into()
            }))
        );
    }

    #[test]
    fn cut_off_and_malformed_streams() {
        let cut = &CHAT[..CHAT.find("data: [DONE]").unwrap()];
        assert!(matches!(
            events(cut, 200, 7).unwrap().last(),
            Some(Err(ProviderError::Malformed(_)))
        ));
        assert!(matches!(
            events("data: {oops\n\n", 200, 7).unwrap().as_slice(),
            [Err(ProviderError::Malformed(_))]
        ));
    }

    #[test]
    fn body_that_breaks_off() {
        let client = FakeHttpClient::new(200, CHAT, 16).failing_after(3);
        let events = collect_body(
            block_on(stream(
                &client,
                &config(),
                &request(),
                ModelLimits::default(),
            ))
            .unwrap(),
        );
        assert_eq!(
            events.last(),
            Some(&Err(ProviderError::Http(HttpError::Body(
                "connection reset".into()
            ))))
        );
    }
}
