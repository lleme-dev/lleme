// SPDX-License-Identifier: AGPL-3.0-only

//! Calendar dates, for prices that take effect on a given day. The core
//! never reads the clock; callers pass the date in.

use std::fmt;
use std::str::FromStr;

/// A day in the proleptic Gregorian calendar, UTC. Ordered by time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date {
    year: u16,
    month: u8,
    day: u8,
}

/// Why a string is not a valid `YYYY-MM-DD` date.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid date {0:?}: expected an existing day written as YYYY-MM-DD")]
pub struct DateError(String);

impl Date {
    /// Builds a date, or returns `None` if the day does not exist.
    pub fn new(year: u16, month: u8, day: u8) -> Option<Date> {
        let leap =
            year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
        let days = match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 if leap => 29,
            2 => 28,
            _ => return None,
        };
        (1..=days)
            .contains(&day)
            .then_some(Date { year, month, day })
    }
}

impl FromStr for Date {
    type Err = DateError;

    fn from_str(s: &str) -> Result<Date, DateError> {
        let err = || DateError(s.to_string());
        let bytes = s.as_bytes();
        let digits = |range: std::ops::Range<usize>| {
            bytes[range.clone()]
                .iter()
                .all(u8::is_ascii_digit)
                .then(|| s[range].parse::<u16>().ok())
                .flatten()
        };
        if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
            return Err(err());
        }
        let (Some(year), Some(month), Some(day)) = (digits(0..4), digits(5..7), digits(8..10))
        else {
            return Err(err());
        };
        Date::new(
            year,
            u8::try_from(month).map_err(|_| err())?,
            u8::try_from(day).map_err(|_| err())?,
        )
        .ok_or_else(err)
    }
}

impl TryFrom<String> for Date {
    type Error = DateError;

    fn try_from(s: String) -> Result<Date, DateError> {
        s.parse()
    }
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_prints_iso_dates() {
        let date: Date = "2026-11-20".parse().unwrap();
        assert_eq!(date, Date::new(2026, 11, 20).unwrap());
        assert_eq!(date.to_string(), "2026-11-20");
    }

    #[test]
    fn rejects_days_that_do_not_exist() {
        assert!("2026-02-29".parse::<Date>().is_err());
        assert!("2028-02-29".parse::<Date>().is_ok());
        assert!("2100-02-29".parse::<Date>().is_err());
        assert!("2000-02-29".parse::<Date>().is_ok());
        assert!("2026-04-31".parse::<Date>().is_err());
        assert!("2026-13-01".parse::<Date>().is_err());
        assert!("2026-00-10".parse::<Date>().is_err());
    }

    #[test]
    fn rejects_other_formats() {
        for s in [
            "2026-1-20",
            "20261120",
            "2026/11/20",
            "2026-11-20T00:00:00Z",
            "+026-11-20",
            "2026-11-2a",
            "",
        ] {
            assert!(s.parse::<Date>().is_err(), "{s} should not parse");
        }
    }

    #[test]
    fn orders_by_time() {
        let d = |s: &str| s.parse::<Date>().unwrap();
        assert!(d("2026-11-20") < d("2026-12-01"));
        assert!(d("2026-12-31") < d("2027-01-01"));
    }
}
