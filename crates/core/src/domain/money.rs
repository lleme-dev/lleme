// SPDX-License-Identifier: AGPL-3.0-only

//! Prices and cost. Money is an integer number of nano-dollars; no
//! floating point is involved anywhere in the calculation.

use serde::Deserialize;

use super::chat::Usage;

/// An amount in nano-dollars (10^-9 USD).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct NanoUsd(pub u64);

/// Token prices in nano-dollars per million tokens. Per-million keeps
/// sub-nano-dollar token prices, such as $0.075 per million, integral.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Price {
    /// Input read without the prompt cache.
    pub input: u64,
    /// Output, including reasoning.
    pub output: u64,
    /// Input read from the prompt cache.
    pub cache_read: u64,
    /// Input written to the short-lived prompt cache.
    pub cache_write_short: u64,
    /// Input written to the long-lived prompt cache.
    pub cache_write_long: u64,
}

/// A higher price that applies to the whole request once its input,
/// cached tokens included, exceeds a threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongContextPrice {
    /// The tier applies when total input tokens are strictly above this.
    pub above_input_tokens: u64,
    /// Prices for such requests.
    pub nano_usd_per_mtok: Price,
}

/// The prices of one model at one point in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriceTiers {
    /// Prices for ordinary requests.
    pub standard: Price,
    /// Prices for long requests, if the model has such a tier.
    pub long_context: Option<LongContextPrice>,
}

impl PriceTiers {
    /// Cost of a request with this usage. Each count is multiplied by its
    /// own price, the products are summed in `u128`, and the total is
    /// rounded once, half up, to whole nano-dollars.
    pub fn cost(&self, usage: &Usage) -> NanoUsd {
        let price = match self.long_context {
            Some(tier) if usage.total_input_tokens() > tier.above_input_tokens => {
                tier.nano_usd_per_mtok
            }
            _ => self.standard,
        };
        let parts = [
            (usage.input_tokens, price.input),
            (usage.output_tokens, price.output),
            (usage.cache_read_tokens, price.cache_read),
            (usage.cache_write_short_tokens, price.cache_write_short),
            (usage.cache_write_long_tokens, price.cache_write_long),
        ];
        let per_mtok: u128 = parts
            .iter()
            .map(|&(tokens, price)| u128::from(tokens) * u128::from(price))
            .sum();
        let rounded = (per_mtok + 500_000) / 1_000_000;
        NanoUsd(u64::try_from(rounded).unwrap_or(u64::MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HAIKU: Price = Price {
        input: 100_000_000,
        output: 500_000_000,
        cache_read: 10_000_000,
        cache_write_short: 125_000_000,
        cache_write_long: 200_000_000,
    };
    const HAIKU_LONG: Price = Price {
        input: 500_000_000,
        output: 2_500_000_000,
        cache_read: 50_000_000,
        cache_write_short: 625_000_000,
        cache_write_long: 1_000_000_000,
    };

    fn tiers() -> PriceTiers {
        PriceTiers {
            standard: HAIKU,
            long_context: Some(LongContextPrice {
                above_input_tokens: 100_000,
                nano_usd_per_mtok: HAIKU_LONG,
            }),
        }
    }

    #[test]
    fn cost_matches_a_hand_computed_value() {
        // 1000 input at $0.10/M = $0.0001; 200 output at $0.50/M = $0.0001;
        // 3000 cache reads at $0.01/M = $0.00003; 400 short writes at
        // $0.125/M = $0.00005; 100 long writes at $0.20/M = $0.00002.
        // Total $0.0003 = 300,000 nano-dollars.
        let usage = Usage {
            input_tokens: 1000,
            output_tokens: 200,
            cache_read_tokens: 3000,
            cache_write_short_tokens: 400,
            cache_write_long_tokens: 100,
        };
        assert_eq!(tiers().cost(&usage), NanoUsd(300_000));
    }

    #[test]
    fn rounds_once_half_up() {
        let one = Price {
            input: 1,
            output: 0,
            cache_read: 0,
            cache_write_short: 0,
            cache_write_long: 0,
        };
        let flat = PriceTiers {
            standard: one,
            long_context: None,
        };
        let input = |n| Usage {
            input_tokens: n,
            ..Usage::default()
        };
        assert_eq!(flat.cost(&input(499_999)), NanoUsd(0));
        assert_eq!(flat.cost(&input(500_000)), NanoUsd(1));
        assert_eq!(flat.cost(&input(1_499_999)), NanoUsd(1));
        // Rounding each part would give 0 + 0; rounding the total gives 1.
        let split = Usage {
            input_tokens: 300_000,
            cache_read_tokens: 300_000,
            ..Usage::default()
        };
        let both = PriceTiers {
            standard: Price {
                cache_read: 1,
                ..one
            },
            long_context: None,
        };
        assert_eq!(both.cost(&split), NanoUsd(1));
    }

    #[test]
    fn long_context_tier_counts_cached_input_and_prices_the_whole_request() {
        let at_threshold = Usage {
            input_tokens: 40_000,
            cache_read_tokens: 60_000,
            output_tokens: 10,
            ..Usage::default()
        };
        // 40,000 x 100 + 60,000 x 10 + 10 x 500 nano-dollars.
        assert_eq!(
            tiers().cost(&at_threshold),
            NanoUsd(4_000_000 + 600_000 + 5_000)
        );
        let over = Usage {
            cache_write_short_tokens: 1,
            ..at_threshold
        };
        // Same counts at the long prices, plus one short write at 625.
        assert_eq!(
            tiers().cost(&over),
            NanoUsd(20_000_000 + 3_000_000 + 25_000 + 625)
        );
    }

    #[test]
    fn huge_usage_saturates_instead_of_wrapping() {
        let usage = Usage {
            output_tokens: u64::MAX,
            ..Usage::default()
        };
        assert_eq!(tiers().cost(&usage), NanoUsd(u64::MAX));
    }
}
