// SPDX-License-Identifier: AGPL-3.0-only

//! The current time. The core never reads the system clock, which panics
//! on wasm32; each runtime implements this port.

use crate::MaybeSync;
use crate::domain::date::Date;

/// A point in time: milliseconds since 1970-01-01T00:00:00Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// Whole seconds since the epoch, as OpenAI's `created` field uses.
    pub fn unix_seconds(self) -> u64 {
        self.0 / 1000
    }

    /// The UTC calendar day this instant falls on.
    pub fn date(self) -> Date {
        Date::from_days_since_epoch(self.0 / 86_400_000)
    }
}

/// Tells the time. Synchronous on both runtimes, so no future.
pub trait Clock: MaybeSync {
    /// The current instant.
    fn now(&self) -> Timestamp;
}

#[cfg(test)]
pub(crate) mod fake {
    use super::{Clock, Timestamp};

    /// A clock stopped at one instant.
    pub(crate) struct FixedClock(pub(crate) Timestamp);

    impl Clock for FixedClock {
        fn now(&self) -> Timestamp {
            self.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::FixedClock;
    use super::*;

    fn date(s: &str) -> Date {
        s.parse().unwrap()
    }

    #[test]
    fn converts_instants_to_utc_days() {
        assert_eq!(Timestamp(0).date(), date("1970-01-01"));
        assert_eq!(Timestamp(86_399_999).date(), date("1970-01-01"));
        assert_eq!(Timestamp(86_400_000).date(), date("1970-01-02"));
        // 2026-11-20T12:41:33.316Z, the start of a recorded measurement.
        assert_eq!(Timestamp(1_795_178_493_316).date(), date("2026-11-20"));
        // Leap days and year ends: 2024-02-29 is day 19782, 2000-12-31 is day 11322.
        assert_eq!(Timestamp(19_782 * 86_400_000).date(), date("2024-02-29"));
        assert_eq!(Timestamp(11_322 * 86_400_000).date(), date("2000-12-31"));
        assert_eq!(Timestamp(11_323 * 86_400_000).date(), date("2001-01-01"));
    }

    #[test]
    fn seconds_and_a_fixed_clock() {
        let clock = FixedClock(Timestamp(1_795_178_493_316));
        assert_eq!(clock.now().unix_seconds(), 1_795_178_493);
    }
}
