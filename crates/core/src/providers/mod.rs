// SPDX-License-Identifier: AGPL-3.0-only

//! Translation between provider formats and the neutral chat types. The
//! network call itself goes through the `HttpClient` port.

pub mod anthropic;
pub mod sse;

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
