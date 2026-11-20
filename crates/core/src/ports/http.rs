// SPDX-License-Identifier: AGPL-3.0-only

//! Outbound HTTP, used by provider adapters to reach provider APIs.

use std::fmt;
use std::future::{Future, poll_fn};
use std::pin::Pin;

use futures_core::Stream;

use crate::{MaybeSend, MaybeSync};

/// A POST request to a provider. Provider chat APIs take only POST.
#[derive(Clone, PartialEq, Eq)]
pub struct HttpRequest {
    /// Absolute URL.
    pub url: String,
    /// Header names and values, in the order they are sent.
    pub headers: Vec<(String, String)>,
    /// Request body, usually JSON.
    pub body: Vec<u8>,
}

/// Headers whose values are credentials and never appear in `Debug`.
const SECRET_HEADERS: [&str; 2] = ["authorization", "x-api-key"];

// Requests carry provider keys; a request written to a log must not.
impl fmt::Debug for HttpRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let headers: Vec<(&str, &str)> = self
            .headers
            .iter()
            .map(|(name, value)| {
                let secret = SECRET_HEADERS.iter().any(|s| name.eq_ignore_ascii_case(s));
                (
                    name.as_str(),
                    if secret { "[redacted]" } else { value.as_str() },
                )
            })
            .collect();
        f.debug_struct("HttpRequest")
            .field("url", &self.url)
            .field("headers", &headers)
            .field("body_len", &self.body.len())
            .finish()
    }
}

/// Reads a whole response body. For non-streamed responses only; streamed
/// ones are processed chunk by chunk.
pub async fn read_to_end<B>(mut body: B) -> Result<Vec<u8>, HttpError>
where
    B: Stream<Item = Result<Vec<u8>, HttpError>> + Unpin,
{
    let mut out = Vec::new();
    while let Some(chunk) = poll_fn(|cx| Pin::new(&mut body).poll_next(cx)).await {
        out.extend_from_slice(&chunk?);
    }
    Ok(out)
}

/// A response whose body arrives as a stream of byte chunks, so a
/// streamed completion can be forwarded before it ends.
#[derive(Debug)]
pub struct HttpResponse<B> {
    /// HTTP status code.
    pub status: u16,
    /// Header names, lowercase, and values.
    pub headers: Vec<(String, String)>,
    /// Body chunks as they arrive. Chunk boundaries carry no meaning.
    pub body: B,
}

/// Failures below the HTTP layer. A non-2xx status is not an error here;
/// the provider adapter interprets it.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum HttpError {
    /// The request could not be sent or no response arrived.
    #[error("http request failed: {0}")]
    Request(String),
    /// The response started but its body could not be read to the end.
    #[error("http response body failed: {0}")]
    Body(String),
}

/// Sends HTTP requests. Implemented by each runtime: Hyper on the server,
/// the Workers fetch API on the edge.
pub trait HttpClient: MaybeSync {
    /// Body stream returned by this client.
    type Body: Stream<Item = Result<Vec<u8>, HttpError>> + MaybeSend + Unpin;

    /// Sends `request` and resolves when the response headers arrive.
    fn send(
        &self,
        request: HttpRequest,
    ) -> impl Future<Output = Result<HttpResponse<Self::Body>, HttpError>> + MaybeSend;
}

#[cfg(test)]
pub(crate) mod fake {
    //! An `HttpClient` that replays a recorded response.

    use std::collections::VecDeque;
    use std::pin::Pin;
    use std::sync::Mutex;
    use std::task::{Context, Poll};

    use futures_core::Stream;

    use super::{HttpClient, HttpError, HttpRequest, HttpResponse};

    /// Body that yields the recorded bytes in chunks of a fixed size, so
    /// tests can split events and UTF-8 characters across chunks.
    pub(crate) struct ChunkedBody(VecDeque<Result<Vec<u8>, HttpError>>);

    impl ChunkedBody {
        /// A body that yields `bytes` in pieces of `chunk` bytes.
        pub(crate) fn of(bytes: &[u8], chunk: usize) -> Self {
            ChunkedBody(bytes.chunks(chunk.max(1)).map(|c| Ok(c.to_vec())).collect())
        }
    }

    impl Stream for ChunkedBody {
        type Item = Result<Vec<u8>, HttpError>;

        fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.0.pop_front())
        }
    }

    /// Replays one recorded response for every request and keeps the
    /// requests it received.
    pub(crate) struct FakeHttpClient {
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
        chunk_size: usize,
        fail_after_chunks: Option<usize>,
        requests: Mutex<Vec<HttpRequest>>,
    }

    impl FakeHttpClient {
        pub(crate) fn new(status: u16, body: impl Into<Vec<u8>>, chunk_size: usize) -> Self {
            Self {
                status,
                headers: Vec::new(),
                body: body.into(),
                chunk_size: chunk_size.max(1),
                fail_after_chunks: None,
                requests: Mutex::new(Vec::new()),
            }
        }

        pub(crate) fn with_header(mut self, name: &str, value: &str) -> Self {
            self.headers
                .push((name.to_ascii_lowercase(), value.to_string()));
            self
        }

        /// Makes the body fail with `HttpError::Body` after `n` chunks.
        pub(crate) fn failing_after(mut self, n: usize) -> Self {
            self.fail_after_chunks = Some(n);
            self
        }

        pub(crate) fn requests(&self) -> Vec<HttpRequest> {
            self.requests.lock().map(|r| r.clone()).unwrap_or_default()
        }
    }

    impl HttpClient for FakeHttpClient {
        type Body = ChunkedBody;

        async fn send(&self, request: HttpRequest) -> Result<HttpResponse<ChunkedBody>, HttpError> {
            if let Ok(mut requests) = self.requests.lock() {
                requests.push(request);
            }
            let mut chunks: VecDeque<_> = self
                .body
                .chunks(self.chunk_size)
                .map(|c| Ok(c.to_vec()))
                .collect();
            if let Some(n) = self.fail_after_chunks {
                chunks.truncate(n);
                chunks.push_back(Err(HttpError::Body("connection reset".into())));
            }
            Ok(HttpResponse {
                status: self.status,
                headers: self.headers.clone(),
                body: ChunkedBody(chunks),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::FakeHttpClient;
    use super::*;
    use crate::test_support::{block_on, collect_body};

    fn request() -> HttpRequest {
        HttpRequest {
            url: "https://provider.test/v1/chat".into(),
            headers: vec![("content-type".into(), "application/json".into())],
            body: b"{}".to_vec(),
        }
    }

    #[test]
    fn fake_replays_the_body_in_chunks_and_keeps_requests() {
        let client =
            FakeHttpClient::new(200, "ol\u{e1} mundo", 3).with_header("Content-Type", "text/plain");
        let response = block_on(client.send(request())).unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(
            response.headers,
            vec![("content-type".to_string(), "text/plain".to_string())]
        );
        let chunks = collect_body(response.body);
        assert_eq!(
            chunks
                .iter()
                .map(|c| c.as_ref().unwrap().len())
                .collect::<Vec<_>>(),
            vec![3, 3, 3, 1]
        );
        let joined: Vec<u8> = chunks.into_iter().flat_map(|c| c.unwrap()).collect();
        assert_eq!(String::from_utf8(joined).unwrap(), "ol\u{e1} mundo");
        assert_eq!(client.requests(), vec![request()]);
    }

    #[test]
    fn debug_output_hides_credentials() {
        let mut req = request();
        req.headers
            .push(("x-api-key".into(), "secret-value".into()));
        req.headers
            .push(("Authorization".into(), "Bearer secret-value".into()));
        let shown = format!("{req:?}");
        assert!(!shown.contains("secret-value"), "{shown}");
        assert!(shown.contains("content-type"));
    }

    #[test]
    fn read_to_end_joins_chunks_and_reports_body_errors() {
        let ok = FakeHttpClient::new(200, "abcdef", 4);
        let body = block_on(ok.send(request())).unwrap().body;
        assert_eq!(block_on(read_to_end(body)), Ok(b"abcdef".to_vec()));
        let broken = FakeHttpClient::new(200, "abcdef", 2).failing_after(1);
        let body = block_on(broken.send(request())).unwrap().body;
        assert_eq!(
            block_on(read_to_end(body)),
            Err(HttpError::Body("connection reset".into()))
        );
    }

    #[test]
    fn fake_body_can_fail_midway() {
        let client = FakeHttpClient::new(200, "abcdef", 2).failing_after(1);
        let response = block_on(client.send(request())).unwrap();
        let chunks = collect_body(response.body);
        assert_eq!(
            chunks,
            vec![
                Ok(b"ab".to_vec()),
                Err(HttpError::Body("connection reset".into()))
            ]
        );
    }

    // Generic code over the port must produce futures Tokio can spawn.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn send_future_is_send_on_native() {
        fn assert_send<T: Send>(value: T) -> T {
            value
        }
        fn call<H: HttpClient>(
            client: &H,
        ) -> impl Future<Output = Result<HttpResponse<H::Body>, HttpError>> + '_ {
            assert_send(client.send(request()))
        }
        let client = FakeHttpClient::new(204, "", 1);
        assert_eq!(block_on(call(&client)).unwrap().status, 204);
    }

    // On wasm32 an adapter holds JavaScript values, which are not Send.
    // This only needs to compile; CI checks the core's tests for wasm32.
    #[cfg(target_arch = "wasm32")]
    mod non_send_client {
        use std::rc::Rc;

        use super::*;
        use crate::ports::http::fake::ChunkedBody;

        #[expect(dead_code, reason = "compiled by the wasm32 check, never constructed")]
        struct RcClient(Rc<()>);

        impl HttpClient for RcClient {
            type Body = ChunkedBody;

            async fn send(&self, _: HttpRequest) -> Result<HttpResponse<ChunkedBody>, HttpError> {
                let _held_across_await = Rc::clone(&self.0);
                Err(HttpError::Request("not run".into()))
            }
        }
    }
}
