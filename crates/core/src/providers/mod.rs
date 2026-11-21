// SPDX-License-Identifier: AGPL-3.0-only

//! Translation between provider formats and the neutral chat types. The
//! network call itself goes through the `HttpClient` port.

pub mod anthropic;
pub mod openai;
pub mod sse;

use serde::Deserialize;

use crate::ports::HttpError;

/// Why a call to a provider did not produce a response.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProviderError {
    /// The request did not reach the provider, or its body broke off.
    #[error(transparent)]
    Http(#[from] HttpError),
    /// The provider answered with an error status.
    #[error("provider returned status {status}: {message}")]
    Status {
        /// HTTP status code.
        status: u16,
        /// The provider's own error code, such as `not_found_error`.
        error_type: Option<String>,
        /// The provider's message, as sent.
        message: String,
    },
    /// The provider reported an error after the stream had started.
    #[error("provider stream was interrupted: {message}")]
    Interrupted {
        /// The provider's own error code, such as `overloaded_error`.
        error_type: Option<String>,
        /// The provider's message, as sent.
        message: String,
    },
    /// The provider answered 2xx with a body this adapter cannot read.
    #[error("provider response is malformed: {0}")]
    Malformed(String),
    /// The request cannot be expressed in this provider's format.
    #[error("request cannot be sent to this provider: {0}")]
    InvalidRequest(String),
}

/// Why a client request cannot be read, or cannot be translated to
/// another provider's format.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RequestError {
    /// A required field is missing or has the wrong type.
    #[error("invalid request: {0}")]
    Invalid(String),
    /// Valid in the client's format, but not translatable yet.
    #[error("request cannot be translated: {0}")]
    Unsupported(String),
}

/// The top-level fields of a client request that the proxy reads before
/// choosing how to send it. Both formats name them the same way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestHead {
    /// Model name as the client asked for it.
    pub model: String,
    /// Whether the client asked for a streamed response.
    pub stream: bool,
    /// OpenAI's `stream_options.include_usage`: whether the client wants
    /// the closing usage chunk. The neutral form does not carry it.
    pub include_usage: bool,
}

#[derive(Deserialize)]
struct RawHead {
    model: String,
    stream: Option<bool>,
    stream_options: Option<RawStreamOptions>,
}

#[derive(Deserialize)]
struct RawStreamOptions {
    include_usage: Option<bool>,
}

/// Reads the [`RequestHead`] of a body in either format. Other fields
/// are scanned, not kept.
pub fn read_head(body: &[u8]) -> Result<RequestHead, RequestError> {
    let raw: RawHead = serde_json::from_slice(body)
        .map_err(|e| RequestError::Invalid(format!("request body: {e}")))?;
    Ok(RequestHead {
        model: raw.model,
        stream: raw.stream.unwrap_or(false),
        include_usage: raw
            .stream_options
            .and_then(|o| o.include_usage)
            .unwrap_or(false),
    })
}
