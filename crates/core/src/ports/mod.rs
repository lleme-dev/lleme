// SPDX-License-Identifier: AGPL-3.0-only

//! Traits through which the core reaches the outside world. Each runtime
//! implements them; the core never does I/O itself.

pub mod clock;
pub mod http;

pub use clock::{Clock, Timestamp};
pub use http::{HttpClient, HttpError, HttpRequest, HttpResponse, read_to_end};
