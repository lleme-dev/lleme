// SPDX-License-Identifier: AGPL-3.0-only

//! The proxy use case: a client request in one provider's format is sent
//! to the provider that serves the model and answered in the client's
//! format, with the usage and cost of the call. When both formats are the
//! same the request and the reply pass through unchanged.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;

use serde_json::Value;

use crate::domain::catalog::{Catalog, Model, PricePeriod, Provider};
use crate::domain::chat::{ChatResponse, StreamEvent, Usage};
use crate::domain::money::{NanoUsd, PriceTiers};
use crate::ports::{Clock, HttpClient, HttpError, read_to_end};
use crate::providers::anthropic::{self, AnthropicConfig, AnthropicEvents, AnthropicStream};
use crate::providers::openai::{self, ModelLimits, OpenAiConfig, OpenAiEvents, OpenAiStream};
use crate::providers::sse::SseParser;
use crate::providers::{ProviderError, RequestError, RequestHead, read_head};

/// The usage and cost of one call, for the trace of the request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    /// Provider that served the request.
    pub provider: Provider,
    /// Model name as the catalog knows it.
    pub model: String,
    /// Token counts the provider reported.
    pub usage: Usage,
    /// Cost at the price in force on the request date; `None` when the
    /// catalog has no price for that date.
    pub cost: Option<NanoUsd>,
}

/// What to send back to the client.
pub enum Reply<B> {
    /// A JSON body, for a whole response or an error.
    Whole {
        /// HTTP status code.
        status: u16,
        /// Body in the client's format.
        body: Vec<u8>,
        /// Present when the provider answered with a response.
        summary: Option<Summary>,
    },
    /// A status 200 response sent as server-sent events. Boxed because
    /// the stream state is several hundred bytes.
    Stream(Box<ReplyStream<B>>),
}

/// Sends client requests to providers. One per process, shared by all
/// requests.
pub struct Proxy<H, C> {
    http: H,
    clock: C,
    catalog: Catalog,
    openai: OpenAiConfig,
    anthropic: AnthropicConfig,
}

impl<H: HttpClient, C: Clock> Proxy<H, C> {
    /// Creates a proxy that reaches both providers through `http`.
    pub fn new(
        http: H,
        clock: C,
        catalog: Catalog,
        openai: OpenAiConfig,
        anthropic: AnthropicConfig,
    ) -> Self {
        Self {
            http,
            clock,
            catalog,
            openai,
            anthropic,
        }
    }

    /// Answers a request whose body is in the `format` of that provider's
    /// API, with the client's request `headers`. Never fails: errors
    /// become a reply in the client's format.
    pub async fn handle(
        &self,
        format: Provider,
        headers: &[(String, String)],
        body: &[u8],
    ) -> Reply<H::Body> {
        let now = self.clock.now();
        let head = match read_head(body) {
            Ok(head) => head,
            Err(error) => return request_error(format, &error),
        };
        // A model missing from the catalog may still exist; the provider
        // of the client's format gets to answer.
        let model = self.catalog.find(&head.model);
        let pricing = Pricing {
            provider: model.map_or(format, |m| m.provider),
            model: head.model.clone(),
            tiers: model
                .and_then(|m| m.price_on(now.date()))
                .map(PricePeriod::tiers),
        };
        match model {
            Some(model) if model.provider != format => {
                self.translate(format, &head, body, model, pricing, now.unix_seconds())
                    .await
            }
            _ => self.forward(format, &head, headers, body, pricing).await,
        }
    }

    async fn translate(
        &self,
        format: Provider,
        head: &RequestHead,
        body: &[u8],
        model: &Model,
        pricing: Pricing,
        created: u64,
    ) -> Reply<H::Body> {
        let parsed = match format {
            Provider::OpenAi => openai::inbound::parse_request(body),
            Provider::Anthropic => anthropic::inbound::parse_request(body),
        };
        let request = match parsed {
            Ok(request) => request,
            Err(error) => return request_error(format, &error),
        };
        let default_max_tokens = u32::try_from(model.max_output_tokens).unwrap_or(u32::MAX);
        let limits = ModelLimits {
            has_tool_reasoning_limit: model.capabilities.has_tool_reasoning_limit,
        };

        if request.stream {
            let source = match model.provider {
                Provider::OpenAi => openai::stream(&self.http, &self.openai, &request, limits)
                    .await
                    .map(Source::OpenAi),
                Provider::Anthropic => {
                    anthropic::stream(&self.http, &self.anthropic, &request, default_max_tokens)
                        .await
                        .map(Source::Anthropic)
                }
            };
            return match source {
                Ok(source) => {
                    let encoder = match format {
                        Provider::OpenAi => Encoder::OpenAi(openai::reply::StreamEncoder::new(
                            created,
                            head.include_usage,
                        )),
                        Provider::Anthropic => {
                            Encoder::Anthropic(anthropic::reply::StreamEncoder::new())
                        }
                    };
                    ReplyStream::reply(format, Flow::Translated { source, encoder }, pricing)
                }
                Err(error) => provider_error(format, &error),
            };
        }

        let response = match model.provider {
            Provider::OpenAi => openai::complete(&self.http, &self.openai, &request, limits).await,
            Provider::Anthropic => {
                anthropic::complete(&self.http, &self.anthropic, &request, default_max_tokens).await
            }
        };
        match response {
            Ok(response) => Reply::Whole {
                status: 200,
                body: render_response(format, &response, created),
                summary: Some(pricing.summary(response.usage)),
            },
            Err(error) => provider_error(format, &error),
        }
    }

    // The client and the provider speak the same format, so the body goes
    // out and comes back as is; usage is read from a copy.
    async fn forward(
        &self,
        format: Provider,
        head: &RequestHead,
        headers: &[(String, String)],
        body: &[u8],
        pricing: Pricing,
    ) -> Reply<H::Body> {
        let request = match format {
            Provider::OpenAi => match openai::forward_request(&self.openai, body, head.stream) {
                Ok(request) => request,
                Err(error) => return request_error(format, &error),
            },
            Provider::Anthropic => {
                anthropic::forward_request(&self.anthropic, body.to_vec(), headers)
            }
        };
        let response = match self.http.send(request).await {
            Ok(response) => response,
            Err(error) => return provider_error(format, &ProviderError::Http(error)),
        };
        let is_success = (200..300).contains(&response.status);
        if head.stream && is_success {
            let copy = match format {
                Provider::OpenAi => UsageReader::OpenAi(OpenAiEvents::new()),
                Provider::Anthropic => UsageReader::Anthropic(AnthropicEvents::new()),
            };
            // The usage chunk was asked for by the gateway, not the client.
            let strip = (format == Provider::OpenAi && !head.include_usage).then(SseParser::new);
            let flow = Flow::Forwarded {
                body: response.body,
                copy,
                strip,
            };
            return ReplyStream::reply(format, flow, pricing);
        }
        let body = match read_to_end(response.body).await {
            Ok(body) => body,
            Err(error) => return provider_error(format, &ProviderError::Http(error)),
        };
        let parsed = match (is_success, format) {
            (false, _) => None,
            (true, Provider::OpenAi) => openai::parse_response(response.status, &body).ok(),
            (true, Provider::Anthropic) => anthropic::parse_response(response.status, &body).ok(),
        };
        Reply::Whole {
            status: response.status,
            body,
            summary: parsed.map(|r| pricing.summary(r.usage)),
        }
    }
}

struct Pricing {
    provider: Provider,
    model: String,
    tiers: Option<PriceTiers>,
}

impl Pricing {
    fn summary(&self, usage: Usage) -> Summary {
        Summary {
            provider: self.provider,
            model: self.model.clone(),
            usage,
            cost: self.tiers.map(|tiers| tiers.cost(&usage)),
        }
    }
}

fn render_response(format: Provider, response: &ChatResponse, created: u64) -> Vec<u8> {
    match format {
        Provider::OpenAi => openai::reply::render_response(response, created),
        Provider::Anthropic => anthropic::reply::render_response(response),
    }
}

fn whole_error<B>(format: Provider, status: u16, error_type: &str, message: &str) -> Reply<B> {
    let body = match format {
        Provider::OpenAi => openai::reply::render_error(error_type, message),
        Provider::Anthropic => anthropic::reply::render_error(error_type, message),
    };
    Reply::Whole {
        status,
        body,
        summary: None,
    }
}

fn request_error<B>(format: Provider, error: &RequestError) -> Reply<B> {
    whole_error(format, 400, "invalid_request_error", &error.to_string())
}

// The provider's status, type and message are kept, so the client sees
// what it would have seen calling the provider. Failures that never got
// a provider answer are a bad gateway; their details stay out of the
// reply, since they can name internal hosts.
fn describe(error: &ProviderError) -> (u16, &str, String) {
    match error {
        ProviderError::Status {
            status,
            error_type,
            message,
        } => (
            *status,
            error_type.as_deref().unwrap_or("api_error"),
            message.clone(),
        ),
        ProviderError::Interrupted {
            error_type,
            message,
        } => (
            502,
            error_type.as_deref().unwrap_or("api_error"),
            message.clone(),
        ),
        ProviderError::InvalidRequest(message) => (400, "invalid_request_error", message.clone()),
        ProviderError::Http(_) => (502, "api_error", "the provider could not be reached".into()),
        ProviderError::Malformed(_) => (
            502,
            "api_error",
            "the provider response could not be read".into(),
        ),
    }
}

fn provider_error<B>(format: Provider, error: &ProviderError) -> Reply<B> {
    let (status, error_type, message) = describe(error);
    whole_error(format, status, error_type, &message)
}

fn stream_error(format: Provider, error: &ProviderError) -> Vec<u8> {
    let (_, error_type, message) = describe(error);
    match format {
        Provider::OpenAi => openai::reply::StreamEncoder::error(error_type, &message),
        Provider::Anthropic => anthropic::reply::StreamEncoder::error(error_type, &message),
    }
}

enum Source<B> {
    OpenAi(OpenAiStream<B>),
    Anthropic(AnthropicStream<B>),
}

enum Encoder {
    OpenAi(openai::reply::StreamEncoder),
    Anthropic(anthropic::reply::StreamEncoder),
}

impl Encoder {
    fn push(&mut self, event: &StreamEvent) -> Vec<u8> {
        match self {
            Encoder::OpenAi(encoder) => encoder.push(event),
            Encoder::Anthropic(encoder) => encoder.push(event),
        }
    }

    fn finish(&mut self) -> Vec<u8> {
        match self {
            Encoder::OpenAi(encoder) => encoder.finish(),
            Encoder::Anthropic(encoder) => encoder.finish(),
        }
    }
}

// Reads a forwarded stream for its usage without changing what is sent.
enum UsageReader {
    OpenAi(OpenAiEvents),
    Anthropic(AnthropicEvents),
}

impl UsageReader {
    fn push(&mut self, bytes: &[u8]) -> Vec<Result<StreamEvent, ProviderError>> {
        match self {
            UsageReader::OpenAi(events) => events.push(bytes),
            UsageReader::Anthropic(events) => events.push(bytes),
        }
    }
}

enum Flow<B> {
    Translated {
        source: Source<B>,
        encoder: Encoder,
    },
    Forwarded {
        body: B,
        copy: UsageReader,
        // Set when OpenAI's closing usage chunk must be removed.
        strip: Option<SseParser>,
    },
}

/// A streamed reply in the client's format, sent as it arrives. An error
/// after the start ends the stream with an error event, since the status
/// has already been sent.
pub struct ReplyStream<B> {
    format: Provider,
    flow: Flow<B>,
    pricing: Pricing,
    usage: Usage,
    is_done: bool,
}

impl<B> ReplyStream<B> {
    fn reply(format: Provider, flow: Flow<B>, pricing: Pricing) -> Reply<B> {
        Reply::Stream(Box::new(ReplyStream {
            format,
            flow,
            pricing,
            usage: Usage::default(),
            is_done: false,
        }))
    }

    /// The usage and cost of the call, once the stream has ended. After an
    /// error it holds the usage reported until then.
    pub fn summary(&self) -> Option<Summary> {
        self.is_done.then(|| self.pricing.summary(self.usage))
    }
}

// Rewrites whole events, leaving out the one that carries only usage.
fn without_usage_chunk(parser: &mut SseParser, bytes: &[u8]) -> Result<Vec<u8>, ProviderError> {
    let mut out = Vec::new();
    for event in parser.push(bytes)? {
        let chunk: Value = serde_json::from_str(&event.data).unwrap_or(Value::Null);
        let is_usage_only =
            chunk["choices"].as_array().is_some_and(Vec::is_empty) && chunk["usage"].is_object();
        if is_usage_only {
            continue;
        }
        for line in event.data.split('\n') {
            out.extend_from_slice(b"data: ");
            out.extend_from_slice(line.as_bytes());
            out.push(b'\n');
        }
        out.push(b'\n');
    }
    Ok(out)
}

impl<B> Stream for ReplyStream<B>
where
    B: Stream<Item = Result<Vec<u8>, HttpError>> + Unpin,
{
    type Item = Vec<u8>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            if this.is_done {
                return Poll::Ready(None);
            }
            let bytes = match &mut this.flow {
                Flow::Translated { source, encoder } => {
                    let next = match source {
                        Source::OpenAi(source) => Pin::new(source).poll_next(cx),
                        Source::Anthropic(source) => Pin::new(source).poll_next(cx),
                    };
                    match next {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Some(Ok(event))) => {
                            if let StreamEvent::Usage(reported) = &event {
                                this.usage = this.usage.merge(*reported);
                            }
                            encoder.push(&event)
                        }
                        Poll::Ready(Some(Err(error))) => {
                            this.is_done = true;
                            stream_error(this.format, &error)
                        }
                        Poll::Ready(None) => {
                            this.is_done = true;
                            encoder.finish()
                        }
                    }
                }
                Flow::Forwarded { body, copy, strip } => match Pin::new(body).poll_next(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Some(Ok(chunk))) => {
                        for event in copy.push(&chunk).into_iter().flatten() {
                            if let StreamEvent::Usage(reported) = event {
                                this.usage = this.usage.merge(reported);
                            }
                        }
                        match strip {
                            None => chunk,
                            Some(parser) => match without_usage_chunk(parser, &chunk) {
                                Ok(bytes) => bytes,
                                Err(error) => {
                                    this.is_done = true;
                                    stream_error(this.format, &error)
                                }
                            },
                        }
                    }
                    Poll::Ready(Some(Err(error))) => {
                        this.is_done = true;
                        stream_error(this.format, &ProviderError::Http(error))
                    }
                    Poll::Ready(None) => {
                        this.is_done = true;
                        Vec::new()
                    }
                },
            };
            if !bytes.is_empty() {
                return Poll::Ready(Some(bytes));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::domain::chat::{ContentPart, FinishReason, ResponseAccumulator, ToolCall};
    use crate::ports::clock::fake::FixedClock;
    use crate::ports::http::fake::{ChunkedBody, FakeHttpClient};
    use crate::ports::{HttpRequest, Timestamp};
    use crate::test_support::{block_on, collect_body};

    // 2026-11-20T12:00:00Z, the day the catalog prices take effect.
    const PRICED: Timestamp = Timestamp(1_795_176_000_000);
    // 2026-11-19T12:00:00Z, before any catalog price.
    const UNPRICED: Timestamp = Timestamp(1_795_089_600_000);

    fn proxy(client: FakeHttpClient, now: Timestamp) -> Proxy<FakeHttpClient, FixedClock> {
        Proxy::new(
            client,
            FixedClock(now),
            Catalog::embedded().unwrap(),
            OpenAiConfig {
                base_url: "https://openai.test".into(),
                api_key: "openai-key".into(),
            },
            AnthropicConfig {
                base_url: "https://anthropic.test".into(),
                api_key: "anthropic-key".into(),
            },
        )
    }

    fn openai_body(model: &str, stream: bool) -> Vec<u8> {
        let mut body = json!({
            "model": model,
            "messages": [{ "role": "user", "content": "Say hello." }],
        });
        if stream {
            body["stream"] = json!(true);
            body["stream_options"] = json!({ "include_usage": true });
        }
        body.to_string().into_bytes()
    }

    fn anthropic_tool_body(model: &str, stream: bool) -> Vec<u8> {
        json!({
            "model": model,
            "max_tokens": 200,
            "stream": stream,
            "messages": [{ "role": "user", "content": "Weather in Lima?" }],
            "tools": [{ "name": "get_weather", "input_schema": { "type": "object" } }],
        })
        .to_string()
        .into_bytes()
    }

    fn whole(reply: Reply<ChunkedBody>) -> (u16, Value, Option<Summary>) {
        match reply {
            Reply::Whole {
                status,
                body,
                summary,
            } => (status, serde_json::from_slice(&body).unwrap(), summary),
            Reply::Stream(_) => panic!("expected a whole reply"),
        }
    }

    fn streamed(reply: Reply<ChunkedBody>) -> (Vec<u8>, Option<Summary>) {
        let Reply::Stream(mut stream) = reply else {
            panic!("expected a stream");
        };
        assert_eq!(stream.summary(), None);
        let bytes = collect_body(&mut stream).concat();
        (bytes, stream.summary())
    }

    fn fold(events: Vec<Result<StreamEvent, ProviderError>>) -> ChatResponse {
        let mut acc = ResponseAccumulator::new();
        events.iter().for_each(|e| acc.push(e.as_ref().unwrap()));
        acc.finish()
    }

    fn sent(client: &FakeHttpClient) -> (String, Value) {
        let request = &client.requests()[0];
        (
            request.url.clone(),
            serde_json::from_slice(&request.body).unwrap(),
        )
    }

    #[test]
    fn openai_client_gets_an_anthropic_response_with_its_cost() {
        let client =
            FakeHttpClient::new(200, include_str!("../../fixtures/anthropic/chat.json"), 64);
        let proxy = proxy(client, PRICED);
        let (status, body, summary) = whole(block_on(proxy.handle(
            Provider::OpenAi,
            &[],
            &openai_body("claude-haiku-5-5", false),
        )));

        assert_eq!(status, 200);
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["created"], 1_795_176_000);
        assert_eq!(
            body["choices"][0]["message"]["content"],
            "Hello from the fixture."
        );
        assert_eq!(body["usage"]["prompt_tokens"], 34);
        let (url, request) = sent(&proxy.http);
        assert_eq!(url, "https://anthropic.test/v1/messages");
        assert_eq!(request["max_tokens"], 128_000);
        // 34 input tokens at 100 nano-USD each, 11 output at 500.
        assert_eq!(
            summary,
            Some(Summary {
                provider: Provider::Anthropic,
                model: "claude-haiku-5-5".into(),
                usage: Usage {
                    input_tokens: 34,
                    output_tokens: 11,
                    ..Usage::default()
                },
                cost: Some(NanoUsd(8_900)),
            })
        );
    }

    #[test]
    fn anthropic_client_gets_an_openai_tool_call() {
        let client = FakeHttpClient::new(200, include_str!("../../fixtures/openai/tool.json"), 64);
        let proxy = proxy(client, PRICED);
        let (status, body, summary) = whole(block_on(proxy.handle(
            Provider::Anthropic,
            &[],
            &anthropic_tool_body("gpt-6-luna", false),
        )));

        assert_eq!(status, 200);
        assert_eq!(body["stop_reason"], "tool_use");
        assert_eq!(body["content"][0]["type"], "tool_use");
        assert_eq!(body["content"][0]["input"], json!({ "city": "Lima" }));
        let (url, request) = sent(&proxy.http);
        assert_eq!(url, "https://openai.test/v1/chat/completions");
        // gpt-6-luna takes tools only without reasoning; see the catalog.
        assert_eq!(request["reasoning_effort"], "none");
        // 135 input tokens at 100 nano-USD each, 18 output at 500.
        assert_eq!(summary.and_then(|s| s.cost), Some(NanoUsd(22_500)));
    }

    #[test]
    fn cost_is_unknown_before_the_first_catalog_price() {
        let client =
            FakeHttpClient::new(200, include_str!("../../fixtures/anthropic/chat.json"), 64);
        let proxy = proxy(client, UNPRICED);
        let (status, _, summary) = whole(block_on(proxy.handle(
            Provider::OpenAi,
            &[],
            &openai_body("claude-haiku-5-5", false),
        )));

        assert_eq!(status, 200);
        let summary = summary.unwrap();
        assert_eq!(summary.usage.output_tokens, 11);
        assert_eq!(summary.cost, None);
    }

    #[test]
    fn openai_client_streams_an_anthropic_response() {
        let client = FakeHttpClient::new(
            200,
            include_str!("../../fixtures/anthropic/chat-stream.sse"),
            7,
        );
        let proxy = proxy(client, PRICED);
        let (bytes, summary) = streamed(block_on(proxy.handle(
            Provider::OpenAi,
            &[],
            &openai_body("claude-haiku-5-5", true),
        )));

        let text = String::from_utf8(bytes.clone()).unwrap();
        assert!(text.ends_with("data: [DONE]\n\n"));
        let events = collect_body(OpenAiStream::new(ChunkedBody::of(&bytes, 5)));
        let response = fold(events);
        assert_eq!(
            response.content,
            vec![ContentPart::Text("Hello from the fixture.".into())]
        );
        assert_eq!(response.usage.output_tokens, 11);
        assert_eq!(summary.and_then(|s| s.cost), Some(NanoUsd(8_900)));
    }

    #[test]
    fn openai_stream_has_no_usage_chunk_unless_asked() {
        let client = FakeHttpClient::new(
            200,
            include_str!("../../fixtures/anthropic/chat-stream.sse"),
            64,
        );
        let proxy = proxy(client, PRICED);
        let body = json!({
            "model": "claude-haiku-5-5",
            "stream": true,
            "messages": [{ "role": "user", "content": "Say hello." }],
        });
        let (bytes, summary) = streamed(block_on(proxy.handle(
            Provider::OpenAi,
            &[],
            body.to_string().as_bytes(),
        )));

        assert!(!String::from_utf8(bytes).unwrap().contains("\"usage\""));
        // The gateway still knows the usage, for the cost.
        assert_eq!(summary.and_then(|s| s.cost), Some(NanoUsd(8_900)));
    }

    #[test]
    fn anthropic_client_streams_an_openai_tool_call() {
        let client = FakeHttpClient::new(
            200,
            include_str!("../../fixtures/openai/tool-stream.sse"),
            7,
        );
        let proxy = proxy(client, PRICED);
        let (bytes, summary) = streamed(block_on(proxy.handle(
            Provider::Anthropic,
            &[],
            &anthropic_tool_body("gpt-6-luna", true),
        )));

        let events = collect_body(AnthropicStream::new(ChunkedBody::of(&bytes, 5)));
        let response = fold(events);
        assert_eq!(
            response.content,
            vec![ContentPart::ToolCall(ToolCall {
                id: "call_fixture02".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"Lima\"}".into(),
            })]
        );
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(summary.map(|s| s.provider), Some(Provider::OpenAi));
    }

    #[test]
    fn stream_cut_by_the_network_ends_with_an_error_event() {
        let client = FakeHttpClient::new(
            200,
            include_str!("../../fixtures/openai/chat-stream.sse"),
            64,
        )
        .failing_after(1);
        let proxy = proxy(client, PRICED);
        let body = anthropic_tool_body("gpt-6-luna", true);
        let (bytes, summary) = streamed(block_on(proxy.handle(Provider::Anthropic, &[], &body)));

        let text = String::from_utf8(bytes).unwrap();
        assert!(text.ends_with(
            "event: error\ndata: {\"error\":{\"message\":\"the provider could not be reached\",\"type\":\"api_error\"},\"type\":\"error\"}\n\n"
        ), "{text}");
        assert!(!text.contains("connection reset"));
        assert!(summary.is_some());
    }

    #[test]
    fn provider_error_status_reaches_the_client_in_its_format() {
        let client = FakeHttpClient::new(
            404,
            include_str!("../../fixtures/anthropic/error-404.json"),
            64,
        );
        let proxy = proxy(client, PRICED);
        let (status, body, summary) = whole(block_on(proxy.handle(
            Provider::OpenAi,
            &[],
            &openai_body("claude-haiku-5-5", true),
        )));

        assert_eq!(status, 404);
        assert_eq!(
            body,
            json!({ "error": {
                "message": "model: lleme-fixture-no-such-model",
                "type": "not_found_error",
                "param": null,
                "code": null,
            } })
        );
        assert_eq!(summary, None);
    }

    #[test]
    fn unreadable_provider_response_is_a_bad_gateway() {
        let client = FakeHttpClient::new(200, "not json", 64);
        let proxy = proxy(client, PRICED);
        let (status, body, _) = whole(block_on(proxy.handle(
            Provider::Anthropic,
            &[],
            &anthropic_tool_body("gpt-6-luna", false),
        )));

        assert_eq!(status, 502);
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "api_error");
    }

    fn header<'a>(request: &'a HttpRequest, name: &str) -> Vec<&'a str> {
        request
            .headers
            .iter()
            .filter(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
            .collect()
    }

    fn raw_whole(reply: Reply<ChunkedBody>) -> (u16, Vec<u8>, Option<Summary>) {
        match reply {
            Reply::Whole {
                status,
                body,
                summary,
            } => (status, body, summary),
            Reply::Stream(_) => panic!("expected a whole reply"),
        }
    }

    // An image, which translation rejects, and spacing a re-encoder would
    // change: only a forwarded body arrives with both.
    const ANTHROPIC_BODY: &[u8] = br#"{"model": "claude-haiku-5-5", "max_tokens": 50, "metadata": {"user_id": "u1"},
        "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "https://example.test/a.png"}}]}]}"#;

    #[test]
    fn anthropic_request_and_response_pass_through_unchanged() {
        let fixture = include_str!("../../fixtures/anthropic/chat.json");
        let proxy = proxy(FakeHttpClient::new(200, fixture, 64), PRICED);
        let headers = [
            ("Anthropic-Version".to_string(), "2023-06-01".to_string()),
            ("anthropic-beta".to_string(), "feature-a".to_string()),
            ("x-api-key".to_string(), "client-key".to_string()),
            ("authorization".to_string(), "Bearer client-key".to_string()),
        ];
        let (status, body, summary) = raw_whole(block_on(proxy.handle(
            Provider::Anthropic,
            &headers,
            ANTHROPIC_BODY,
        )));

        assert_eq!((status, body.as_slice()), (200, fixture.as_bytes()));
        let request = &proxy.http.requests()[0];
        assert_eq!(request.url, "https://anthropic.test/v1/messages");
        assert_eq!(request.body, ANTHROPIC_BODY);
        assert_eq!(header(request, "x-api-key"), ["anthropic-key"]);
        assert_eq!(header(request, "anthropic-version"), ["2023-06-01"]);
        assert_eq!(header(request, "anthropic-beta"), ["feature-a"]);
        assert!(header(request, "authorization").is_empty());
        assert_eq!(summary.and_then(|s| s.cost), Some(NanoUsd(8_900)));
    }

    #[test]
    fn openai_request_and_response_pass_through_unchanged() {
        let fixture = include_str!("../../fixtures/openai/tool.json");
        let proxy = proxy(FakeHttpClient::new(200, fixture, 64), PRICED);
        let request_body = br#"{"model": "gpt-6-luna", "logprobs": true,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}]}]}"#;
        let headers = [("authorization".to_string(), "Bearer client-key".to_string())];
        let (status, body, summary) = raw_whole(block_on(proxy.handle(
            Provider::OpenAi,
            &headers,
            request_body,
        )));

        assert_eq!((status, body.as_slice()), (200, fixture.as_bytes()));
        let request = &proxy.http.requests()[0];
        assert_eq!(request.body, request_body);
        assert_eq!(header(request, "authorization"), ["Bearer openai-key"]);
        assert_eq!(summary.and_then(|s| s.cost), Some(NanoUsd(22_500)));
    }

    #[test]
    fn model_missing_from_the_catalog_goes_to_the_client_format_provider() {
        let fixture = include_str!("../../fixtures/openai/chat.json");
        let proxy = proxy(FakeHttpClient::new(200, fixture, 64), PRICED);
        let (status, _, summary) = raw_whole(block_on(proxy.handle(
            Provider::OpenAi,
            &[],
            &openai_body("gpt-not-in-catalog", false),
        )));

        assert_eq!(status, 200);
        assert_eq!(
            proxy.http.requests()[0].url,
            "https://openai.test/v1/chat/completions"
        );
        assert_eq!(
            summary,
            Some(Summary {
                provider: Provider::OpenAi,
                model: "gpt-not-in-catalog".into(),
                usage: Usage {
                    input_tokens: 20,
                    output_tokens: 8,
                    ..Usage::default()
                },
                cost: None,
            })
        );
    }

    #[test]
    fn forwarded_error_status_passes_through_unchanged() {
        let fixture = include_str!("../../fixtures/anthropic/error-404.json");
        let proxy = proxy(FakeHttpClient::new(404, fixture, 64), PRICED);
        let (status, body, summary) = raw_whole(block_on(proxy.handle(
            Provider::Anthropic,
            &[],
            ANTHROPIC_BODY,
        )));

        assert_eq!((status, body.as_slice()), (404, fixture.as_bytes()));
        assert_eq!(summary, None);
    }

    #[test]
    fn openai_stream_passes_through_when_the_client_asks_for_usage() {
        let fixture = include_str!("../../fixtures/openai/chat-stream.sse");
        let proxy = proxy(FakeHttpClient::new(200, fixture, 7), PRICED);
        let request_body = openai_body("gpt-6-luna", true);
        let (bytes, summary) =
            streamed(block_on(proxy.handle(Provider::OpenAi, &[], &request_body)));

        assert_eq!(bytes, fixture.as_bytes());
        assert_eq!(proxy.http.requests()[0].body, request_body);
        // 20 input tokens at 100 nano-USD each, 8 output at 500.
        assert_eq!(summary.and_then(|s| s.cost), Some(NanoUsd(6_000)));
    }

    #[test]
    fn openai_stream_usage_chunk_is_removed_when_the_client_did_not_ask() {
        let fixture = include_str!("../../fixtures/openai/chat-stream.sse");
        let proxy = proxy(FakeHttpClient::new(200, fixture, 7), PRICED);
        let request_body = json!({
            "model": "gpt-6-luna",
            "stream": true,
            "stream_options": { "include_obfuscation": false },
            "messages": [{ "role": "user", "content": "Say hello." }],
        });
        let (bytes, summary) = streamed(block_on(proxy.handle(
            Provider::OpenAi,
            &[],
            request_body.to_string().as_bytes(),
        )));

        let (_, sent) = sent(&proxy.http);
        assert_eq!(
            sent["stream_options"],
            json!({ "include_obfuscation": false, "include_usage": true })
        );
        let text = String::from_utf8(bytes.clone()).unwrap();
        assert!(!text.contains("\"choices\":[]"), "{text}");
        assert!(text.ends_with("data: [DONE]\n\n"));
        let response = fold(collect_body(OpenAiStream::new(ChunkedBody::of(&bytes, 5))));
        assert_eq!(
            response.content,
            vec![ContentPart::Text("Hello from the fixture.".into())]
        );
        assert_eq!(summary.and_then(|s| s.cost), Some(NanoUsd(6_000)));
    }

    #[test]
    fn anthropic_stream_passes_through_unchanged() {
        let fixture = include_str!("../../fixtures/anthropic/chat-stream.sse");
        let proxy = proxy(FakeHttpClient::new(200, fixture, 7), PRICED);
        let request_body =
            br#"{"model": "claude-haiku-5-5", "max_tokens": 50, "stream": true, "messages": []}"#;
        let (bytes, summary) = streamed(block_on(proxy.handle(
            Provider::Anthropic,
            &[],
            request_body,
        )));

        assert_eq!(bytes, fixture.as_bytes());
        assert_eq!(summary.and_then(|s| s.cost), Some(NanoUsd(8_900)));
    }

    #[test]
    fn forwarded_stream_cut_by_the_network_ends_with_an_error_event() {
        let fixture = include_str!("../../fixtures/anthropic/chat-stream.sse");
        let client = FakeHttpClient::new(200, fixture, 64).failing_after(1);
        let proxy = proxy(client, PRICED);
        let request_body =
            br#"{"model": "claude-haiku-5-5", "max_tokens": 50, "stream": true, "messages": []}"#;
        let (bytes, summary) = streamed(block_on(proxy.handle(
            Provider::Anthropic,
            &[],
            request_body,
        )));

        let text = String::from_utf8(bytes).unwrap();
        assert!(text.starts_with(&fixture[..64]));
        assert!(text.ends_with(
            "event: error\ndata: {\"error\":{\"message\":\"the provider could not be reached\",\"type\":\"api_error\"},\"type\":\"error\"}\n\n"
        ), "{text}");
        assert!(summary.is_some());
    }

    #[test]
    fn unreadable_client_request_is_a_bad_request() {
        let client = FakeHttpClient::new(200, "", 64);
        let proxy = proxy(client, PRICED);
        let (status, body, _) = whole(block_on(proxy.handle(
            Provider::OpenAi,
            &[],
            b"{\"model\": 1}",
        )));

        assert_eq!(status, 400);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(proxy.http.requests().is_empty());
    }
}
