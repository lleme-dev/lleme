// SPDX-License-Identifier: AGPL-3.0-only

//! Thread-safety bounds that hold on Tokio and vanish on Workers.
//!
//! Tokio moves futures across threads, so native adapters need `Send` and
//! `Sync`. A Worker is single-threaded and its JavaScript values are not
//! `Send`, so the same bounds would make every Workers adapter impossible.
//! Ports use these traits instead of writing `Send` or `Sync` directly.

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    /// `Send` on native targets; no bound on wasm32.
    pub trait MaybeSend: Send {}
    impl<T: Send + ?Sized> MaybeSend for T {}

    /// `Sync` on native targets; no bound on wasm32.
    pub trait MaybeSync: Sync {}
    impl<T: Sync + ?Sized> MaybeSync for T {}
}

#[cfg(target_arch = "wasm32")]
mod imp {
    /// `Send` on native targets; no bound on wasm32.
    pub trait MaybeSend {}
    impl<T: ?Sized> MaybeSend for T {}

    /// `Sync` on native targets; no bound on wasm32.
    pub trait MaybeSync {}
    impl<T: ?Sized> MaybeSync for T {}
}

pub use imp::{MaybeSend, MaybeSync};
