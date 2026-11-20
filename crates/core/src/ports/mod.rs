// SPDX-License-Identifier: AGPL-3.0-only

//! Traits through which the core reaches the outside world. Each runtime
//! implements them; the core never does I/O itself.

pub mod http;

pub use http::{HttpClient, HttpError, HttpRequest, HttpResponse};
