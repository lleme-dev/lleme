// SPDX-License-Identifier: AGPL-3.0-only

//! Helpers for unit tests in the core. The fakes never return `Pending`,
//! so a polling loop is enough and the core needs no async runtime.

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use futures_core::Stream;

pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    match future.as_mut().poll(&mut cx) {
        Poll::Ready(output) => output,
        Poll::Pending => panic!("test future returned Pending; the fakes never should"),
    }
}

pub(crate) fn collect_body<S: Stream + Unpin>(mut body: S) -> Vec<S::Item> {
    let mut cx = Context::from_waker(Waker::noop());
    let mut items = Vec::new();
    loop {
        match std::pin::Pin::new(&mut body).poll_next(&mut cx) {
            Poll::Ready(Some(item)) => items.push(item),
            Poll::Ready(None) => return items,
            Poll::Pending => panic!("test stream returned Pending; the fakes never should"),
        }
    }
}
