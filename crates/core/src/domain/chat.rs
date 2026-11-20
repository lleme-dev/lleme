// SPDX-License-Identifier: AGPL-3.0-only

//! Chat requests, responses and stream events in a provider-neutral form.
//!
//! Adapters translate each provider format to and from these types, so a
//! request in one format can reach a provider that speaks another, and the
//! use cases read usage and content without knowing either format.

use std::collections::HashMap;

/// A chat completion request.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    /// Model name as the client asked for it.
    pub model: String,
    /// System prompt. OpenAI system and developer messages end up here.
    pub system: Option<String>,
    /// Conversation turns, oldest first.
    pub messages: Vec<Message>,
    /// Tools the model may call.
    pub tools: Vec<ToolDefinition>,
    /// How the model may use the tools. `None` leaves it to the provider.
    pub tool_choice: Option<ToolChoice>,
    /// Upper bound on output tokens.
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f64>,
    /// Sequences that stop generation.
    pub stop: Vec<String>,
    /// Whether the client asked for a streamed response.
    pub stream: bool,
}

/// Who wrote a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The client side, including tool results.
    User,
    /// The model.
    Assistant,
}

/// One conversation turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// Author of the turn.
    pub role: Role,
    /// Parts in order.
    pub content: Vec<ContentPart>,
}

/// A piece of a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentPart {
    /// Plain text.
    Text(String),
    /// A call the model made to a tool.
    ToolCall(ToolCall),
    /// The client's answer to a tool call.
    ToolResult {
        /// Id of the call this answers.
        call_id: String,
        /// Result as text.
        content: String,
        /// Whether the tool failed.
        is_error: bool,
    },
}

/// A tool call made by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Provider-assigned call id.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Arguments as JSON text. Kept as text because both providers stream
    /// arguments as text fragments.
    pub arguments: String,
}

/// A tool the model may call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// What the tool does, for the model.
    pub description: Option<String>,
    /// JSON Schema of the arguments.
    pub parameters: serde_json::Value,
}

/// How the model may use the tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model decides.
    Auto,
    /// The model must not call a tool.
    None,
    /// The model must call some tool.
    Required,
    /// The model must call this tool.
    Tool(String),
}

/// Why generation ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// Natural end or a stop sequence.
    Stop,
    /// The output token limit was reached.
    Length,
    /// The model stopped to call tools.
    ToolCalls,
    /// The provider withheld or cut the output.
    ContentFilter,
    /// A reason this version does not know, as the provider sent it.
    Other(String),
}

/// Token counts reported by the provider.
///
/// `input_tokens` excludes cached tokens, so cost is each count times its
/// own price. OpenAI reports cached tokens inside its prompt count; its
/// adapter subtracts them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Input tokens read without the provider's prompt cache.
    pub input_tokens: u64,
    /// Output tokens, including reasoning tokens.
    pub output_tokens: u64,
    /// Input tokens read from the prompt cache.
    pub cache_read_tokens: u64,
    /// Input tokens written to the prompt cache.
    pub cache_write_tokens: u64,
}

impl Usage {
    /// Combines two reports of the same response. Providers send running
    /// totals, sometimes repeating a count with zero, so each field keeps
    /// the larger value.
    pub fn merge(self, other: Usage) -> Usage {
        Usage {
            input_tokens: self.input_tokens.max(other.input_tokens),
            output_tokens: self.output_tokens.max(other.output_tokens),
            cache_read_tokens: self.cache_read_tokens.max(other.cache_read_tokens),
            cache_write_tokens: self.cache_write_tokens.max(other.cache_write_tokens),
        }
    }
}

/// A complete, non-streamed response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatResponse {
    /// Provider response id.
    pub id: String,
    /// Model that served the request, as the provider names it.
    pub model: String,
    /// Text and tool calls, in order.
    pub content: Vec<ContentPart>,
    /// Why generation ended; `None` if the provider never said.
    pub finish_reason: Option<FinishReason>,
    /// Token counts.
    pub usage: Usage,
}

/// One event of a streamed response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// The response began.
    Start {
        /// Provider response id.
        id: String,
        /// Model that serves the request.
        model: String,
    },
    /// More output text.
    TextDelta(String),
    /// The model began a tool call. `index` identifies the call in later
    /// deltas of the same response.
    ToolCallStart {
        /// Position of the call among the response's tool calls.
        index: u32,
        /// Provider-assigned call id.
        id: String,
        /// Tool name.
        name: String,
    },
    /// More argument text for the tool call with this `index`.
    ToolCallDelta {
        /// Position given in `ToolCallStart`.
        index: u32,
        /// JSON text fragment.
        arguments: String,
    },
    /// Token counts so far; see [`Usage::merge`].
    Usage(Usage),
    /// Why generation ended.
    Finish(FinishReason),
}

/// Folds stream events into the response a non-streamed call would give.
#[derive(Debug, Default)]
pub struct ResponseAccumulator {
    id: String,
    model: String,
    content: Vec<ContentPart>,
    tool_positions: HashMap<u32, usize>,
    finish_reason: Option<FinishReason>,
    usage: Usage,
}

impl ResponseAccumulator {
    /// Creates an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies one event. A delta for a tool call that never started is
    /// dropped, since there is no call to attach it to.
    pub fn push(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::Start { id, model } => {
                self.id.clone_from(id);
                self.model.clone_from(model);
            }
            StreamEvent::TextDelta(text) => match self.content.last_mut() {
                Some(ContentPart::Text(existing)) => existing.push_str(text),
                _ => self.content.push(ContentPart::Text(text.clone())),
            },
            StreamEvent::ToolCallStart { index, id, name } => {
                self.tool_positions.insert(*index, self.content.len());
                self.content.push(ContentPart::ToolCall(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                }));
            }
            StreamEvent::ToolCallDelta { index, arguments } => {
                let part = self
                    .tool_positions
                    .get(index)
                    .and_then(|&at| self.content.get_mut(at));
                if let Some(ContentPart::ToolCall(call)) = part {
                    call.arguments.push_str(arguments);
                }
            }
            StreamEvent::Usage(usage) => self.usage = self.usage.merge(*usage),
            StreamEvent::Finish(reason) => self.finish_reason = Some(reason.clone()),
        }
    }

    /// Returns the response built so far.
    pub fn finish(self) -> ChatResponse {
        ChatResponse {
            id: self.id,
            model: self.model,
            content: self.content,
            finish_reason: self.finish_reason,
            usage: self.usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(input: u64, output: u64) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            ..Usage::default()
        }
    }

    fn fold(events: &[StreamEvent]) -> ChatResponse {
        let mut acc = ResponseAccumulator::new();
        for event in events {
            acc.push(event);
        }
        acc.finish()
    }

    #[test]
    fn text_deltas_join_into_one_part() {
        let response = fold(&[
            StreamEvent::Start {
                id: "r1".into(),
                model: "m".into(),
            },
            StreamEvent::TextDelta("ol".into()),
            StreamEvent::TextDelta("\u{e1}".into()),
            StreamEvent::Finish(FinishReason::Stop),
        ]);
        assert_eq!(response.id, "r1");
        assert_eq!(response.model, "m");
        assert_eq!(response.content, vec![ContentPart::Text("ol\u{e1}".into())]);
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn tool_call_arguments_join_by_index_and_keep_order() {
        let response = fold(&[
            StreamEvent::TextDelta("Checking.".into()),
            StreamEvent::ToolCallStart {
                index: 0,
                id: "c0".into(),
                name: "weather".into(),
            },
            StreamEvent::ToolCallStart {
                index: 1,
                id: "c1".into(),
                name: "time".into(),
            },
            StreamEvent::ToolCallDelta {
                index: 0,
                arguments: "{\"city\":".into(),
            },
            StreamEvent::ToolCallDelta {
                index: 1,
                arguments: "{}".into(),
            },
            StreamEvent::ToolCallDelta {
                index: 0,
                arguments: "\"Lima\"}".into(),
            },
            StreamEvent::ToolCallDelta {
                index: 7,
                arguments: "lost".into(),
            },
            StreamEvent::Finish(FinishReason::ToolCalls),
        ]);
        let call = |id: &str, name: &str, arguments: &str| {
            ContentPart::ToolCall(ToolCall {
                id: id.into(),
                name: name.into(),
                arguments: arguments.into(),
            })
        };
        assert_eq!(
            response.content,
            vec![
                ContentPart::Text("Checking.".into()),
                call("c0", "weather", "{\"city\":\"Lima\"}"),
                call("c1", "time", "{}"),
            ]
        );
    }

    #[test]
    fn text_after_a_tool_call_starts_a_new_part() {
        let response = fold(&[
            StreamEvent::ToolCallStart {
                index: 0,
                id: "c0".into(),
                name: "t".into(),
            },
            StreamEvent::TextDelta("after".into()),
        ]);
        assert_eq!(response.content.len(), 2);
        assert_eq!(response.content[1], ContentPart::Text("after".into()));
    }

    #[test]
    fn usage_reports_merge_field_by_field() {
        let response = fold(&[
            StreamEvent::Usage(Usage {
                cache_read_tokens: 9,
                ..usage(12, 1)
            }),
            StreamEvent::Usage(usage(0, 40)),
        ]);
        assert_eq!(
            response.usage,
            Usage {
                cache_read_tokens: 9,
                ..usage(12, 40)
            }
        );
    }

    #[test]
    fn missing_finish_stays_unknown() {
        assert_eq!(
            fold(&[StreamEvent::TextDelta("cut".into())]).finish_reason,
            None
        );
    }
}
