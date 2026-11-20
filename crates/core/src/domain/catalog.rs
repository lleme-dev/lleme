// SPDX-License-Identifier: AGPL-3.0-only

//! The model catalog: which models exist, what they can do and what they
//! cost on a given date. The data lives in `crates/core/data/models.json`
//! and is compiled into the core, so both runtimes read the same prices.

use serde::Deserialize;

use super::date::Date;
use super::money::{LongContextPrice, Price, PriceTiers};

const EMBEDDED: &str = include_str!("../../data/models.json");

/// Who serves a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
pub enum Provider {
    /// Anthropic Messages API.
    #[serde(rename = "anthropic")]
    Anthropic,
    /// OpenAI Chat Completions API.
    #[serde(rename = "openai")]
    OpenAi,
}

/// What a model accepts, for routing filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capabilities {
    /// Accepts tool definitions and returns tool calls.
    pub tools: bool,
    /// Accepts images in the input.
    pub image_input: bool,
}

/// Prices in force from one date until the next period starts.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PricePeriod {
    /// First day these prices apply, UTC.
    #[serde(deserialize_with = "date_from_string")]
    pub effective_from: Date,
    /// Where the prices were read, for review.
    pub source: String,
    /// Standard prices.
    pub nano_usd_per_mtok: Price,
    /// Higher prices for long requests, if any.
    #[serde(default)]
    pub long_context: Option<LongContextPrice>,
}

impl PricePeriod {
    /// The prices of this period in the form cost is computed from.
    pub fn tiers(&self) -> PriceTiers {
        PriceTiers {
            standard: self.nano_usd_per_mtok,
            long_context: self.long_context,
        }
    }
}

/// One model as a provider names it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    /// Provider that serves it.
    pub provider: Provider,
    /// Model name in provider requests.
    pub name: String,
    /// Maximum input plus output tokens.
    pub context_window: u64,
    /// Maximum output tokens.
    pub max_output_tokens: u64,
    /// What the model accepts.
    pub capabilities: Capabilities,
    /// Price history, oldest first.
    pub prices: Vec<PricePeriod>,
}

impl Model {
    /// Prices in force on `date`, or `None` before the first period.
    pub fn price_on(&self, date: Date) -> Option<&PricePeriod> {
        self.prices.iter().rev().find(|p| p.effective_from <= date)
    }
}

/// A catalog file that cannot be used.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CatalogError {
    /// The file is not valid JSON of the expected shape.
    #[error("model catalog is malformed: {0}")]
    Malformed(String),
    /// The same provider and name appear twice.
    #[error("model catalog lists {0} twice")]
    Duplicate(String),
    /// A model has no prices, or periods out of date order.
    #[error("model catalog prices for {0} are missing or not in date order")]
    PriceOrder(String),
}

/// All known models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Catalog {
    models: Vec<Model>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogFile {
    models: Vec<Model>,
}

fn date_from_string<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Date, D::Error> {
    let text = String::deserialize(deserializer)?;
    text.parse().map_err(serde::de::Error::custom)
}

impl Catalog {
    /// The catalog shipped with this build.
    pub fn embedded() -> Result<Catalog, CatalogError> {
        Catalog::from_json(EMBEDDED)
    }

    /// Parses and checks a catalog file.
    pub fn from_json(json: &str) -> Result<Catalog, CatalogError> {
        let file: CatalogFile =
            serde_json::from_str(json).map_err(|e| CatalogError::Malformed(e.to_string()))?;
        let mut seen = std::collections::HashSet::new();
        for model in &file.models {
            let label = format!("{:?} {}", model.provider, model.name);
            if !seen.insert((model.provider, model.name.as_str())) {
                return Err(CatalogError::Duplicate(label));
            }
            let ordered = model
                .prices
                .windows(2)
                .all(|w| w[0].effective_from < w[1].effective_from);
            if model.prices.is_empty() || !ordered {
                return Err(CatalogError::PriceOrder(label));
            }
        }
        Ok(Catalog {
            models: file.models,
        })
    }

    /// The model a provider serves under `name`.
    pub fn model(&self, provider: Provider, name: &str) -> Option<&Model> {
        self.models
            .iter()
            .find(|m| m.provider == provider && m.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::Usage;
    use crate::domain::money::NanoUsd;

    fn date(s: &str) -> Date {
        s.parse().unwrap()
    }

    #[test]
    fn embedded_catalog_loads_and_prices_a_recorded_usage() {
        let catalog = Catalog::embedded().unwrap();
        let haiku = catalog
            .model(Provider::Anthropic, "claude-haiku-5-5")
            .unwrap();
        assert!(haiku.capabilities.tools);
        // Usage recorded from a real Claude Haiku 5.5 response: 34 in, 11 out.
        // 34 x $0.10/M + 11 x $0.50/M = $0.0000089.
        let usage = Usage {
            input_tokens: 34,
            output_tokens: 11,
            ..Usage::default()
        };
        let price = haiku.price_on(date("2026-11-20")).unwrap();
        assert_eq!(price.tiers().cost(&usage), NanoUsd(8_900));
        // A recorded 1-hour cache write: 32 in, 11 out, 5403 written at $0.20/M.
        let long_write = Usage {
            input_tokens: 32,
            output_tokens: 11,
            cache_write_long_tokens: 5403,
            ..Usage::default()
        };
        assert_eq!(
            price.tiers().cost(&long_write),
            NanoUsd(3_200 + 5_500 + 1_080_600)
        );
        assert!(catalog.model(Provider::OpenAi, "gpt-6-luna").is_some());
        assert!(
            catalog
                .model(Provider::OpenAi, "claude-haiku-5-5")
                .is_none()
        );
    }

    fn model(name: &str, prices: &[String]) -> String {
        format!(
            r#"{{"provider":"openai","name":"{name}","context_window":10,"max_output_tokens":5,
            "capabilities":{{"tools":false,"image_input":false}},"prices":[{}]}}"#,
            prices.join(",")
        )
    }

    fn file(models: &[String]) -> String {
        format!(r#"{{"models":[{}]}}"#, models.join(","))
    }

    fn period(from: &str, input: u64) -> String {
        format!(
            r#"{{"effective_from":"{from}","source":"test","nano_usd_per_mtok":{{"input":{input},"output":0,
            "cache_read":0,"cache_write_short":0,"cache_write_long":0}}}}"#
        )
    }

    #[test]
    fn picks_the_period_in_force_on_the_date() {
        let json = file(&[model(
            "m",
            &[period("2026-01-01", 1), period("2026-06-01", 2)],
        )]);
        let catalog = Catalog::from_json(&json).unwrap();
        let model = catalog.model(Provider::OpenAi, "m").unwrap();
        assert_eq!(model.price_on(date("2025-12-31")), None);
        assert_eq!(
            model
                .price_on(date("2026-01-01"))
                .unwrap()
                .nano_usd_per_mtok
                .input,
            1
        );
        assert_eq!(
            model
                .price_on(date("2026-05-31"))
                .unwrap()
                .nano_usd_per_mtok
                .input,
            1
        );
        assert_eq!(
            model
                .price_on(date("2026-06-01"))
                .unwrap()
                .nano_usd_per_mtok
                .input,
            2
        );
    }

    #[test]
    fn rejects_unusable_files() {
        let error = |json: String| Catalog::from_json(&json).unwrap_err();
        let unordered = model("m", &[period("2026-06-01", 2), period("2026-01-01", 1)]);
        assert!(matches!(
            error(file(&[unordered])),
            CatalogError::PriceOrder(_)
        ));
        assert!(matches!(
            error(file(&[model("m", &[])])),
            CatalogError::PriceOrder(_)
        ));
        let ok = model("m", &[period("2026-01-01", 1)]);
        assert!(matches!(
            error(file(&[ok.clone(), ok.clone()])),
            CatalogError::Duplicate(_)
        ));
        assert!(matches!(
            error(file(&[model("m", &[period("2026-02-30", 1)])])),
            CatalogError::Malformed(_)
        ));
        assert!(matches!(
            error(file(&[ok.replace("\"tools\"", "\"tool\"")])),
            CatalogError::Malformed(_)
        ));
    }
}
