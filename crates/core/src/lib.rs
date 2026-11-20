// SPDX-License-Identifier: AGPL-3.0-only

//! Domain types, use cases, ports and provider format translation for Lleme.

mod bounds;
pub mod ports;

pub use bounds::{MaybeSend, MaybeSync};

#[cfg(test)]
mod test_support;
