// SPDX-License-Identifier: AGPL-3.0-only

//! Server-Sent Events, parsed as bytes arrive. Chunk boundaries carry no
//! meaning: an event, a line or a UTF-8 character may be split across
//! chunks, so text is decoded only once a whole line is buffered.

use super::ProviderError;

/// A line longer than this is treated as a broken stream rather than
/// buffered without bound. Provider events are a few kilobytes.
const MAX_LINE_BYTES: usize = 1 << 20;

/// One dispatched event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field, if the event had one.
    pub event: Option<String>,
    /// The `data:` lines, joined with newlines.
    pub data: String,
}

/// Incremental parser for a `text/event-stream` body.
#[derive(Debug, Default)]
pub struct SseParser {
    pending: Vec<u8>,
    event: Option<String>,
    data: Option<String>,
}

impl SseParser {
    /// Creates an empty parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds the next bytes and returns every event they complete.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<SseEvent>, ProviderError> {
        self.pending.extend_from_slice(bytes);
        let mut events = Vec::new();
        let mut start = 0;
        while let Some(offset) = self.pending[start..]
            .iter()
            .position(|&b| b == b'\n' || b == b'\r')
        {
            let end = start + offset;
            // A trailing '\r' may be the first half of "\r\n"; wait for more.
            let width = match (self.pending[end], self.pending.get(end + 1)) {
                (b'\r', Some(b'\n')) => 2,
                (b'\r', None) => break,
                _ => 1,
            };
            let line = std::str::from_utf8(&self.pending[start..end])
                .map_err(|_| ProviderError::Malformed("event stream is not valid UTF-8".into()))?
                .to_string();
            start = end + width;
            if let Some(event) = self.line(&line) {
                events.push(event);
            }
        }
        self.pending.drain(..start);
        if self.pending.len() > MAX_LINE_BYTES {
            return Err(ProviderError::Malformed(
                "event stream line is too long".into(),
            ));
        }
        Ok(events)
    }

    /// Whether bytes or fields are buffered that no blank line has closed.
    /// True at the end of a body means the stream was cut off.
    pub fn has_pending(&self) -> bool {
        self.data.is_some() || self.pending.iter().any(|b| !b.is_ascii_whitespace())
    }

    fn line(&mut self, line: &str) -> Option<SseEvent> {
        if line.is_empty() {
            let event = self.event.take();
            return self.data.take().map(|data| SseEvent { event, data });
        }
        if line.starts_with(':') {
            return None;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "event" => self.event = Some(value.to_string()),
            "data" => match &mut self.data {
                Some(data) => {
                    data.push('\n');
                    data.push_str(value);
                }
                None => self.data = Some(value.to_string()),
            },
            // `id` and `retry` matter to browsers reconnecting, not here.
            _ => {}
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_in_pieces(input: &[u8], piece: usize) -> (Vec<SseEvent>, SseParser) {
        let mut parser = SseParser::new();
        let mut events = Vec::new();
        for chunk in input.chunks(piece) {
            events.extend(parser.push(chunk).unwrap());
        }
        (events, parser)
    }

    #[test]
    fn recorded_anthropic_stream_parses_the_same_in_any_chunking() {
        let fixture = include_bytes!("../../fixtures/anthropic/chat-stream.sse");
        let (whole, _) = parse_in_pieces(fixture, fixture.len());
        let names: Vec<_> = whole
            .iter()
            .map(|e| e.event.as_deref().unwrap_or(""))
            .collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "ping",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        for piece in [1, 2, 7, 64] {
            let (events, parser) = parse_in_pieces(fixture, piece);
            assert_eq!(events, whole, "piece size {piece}");
            assert!(!parser.has_pending());
        }
    }

    #[test]
    fn recorded_openai_stream_has_data_only_events() {
        let fixture = include_bytes!("../../fixtures/openai/chat-stream.sse");
        let (events, _) = parse_in_pieces(fixture, 7);
        assert_eq!(events.len(), 9);
        assert!(events.iter().all(|e| e.event.is_none()));
        assert_eq!(events.last().map(|e| e.data.as_str()), Some("[DONE]"));
    }

    #[test]
    fn utf8_split_across_chunks_survives() {
        let input = "data: ol\u{e1}\n\n".as_bytes();
        let split = input.iter().position(|&b| b == 0xc3).unwrap() + 1;
        let mut parser = SseParser::new();
        assert_eq!(parser.push(&input[..split]).unwrap(), []);
        let events = parser.push(&input[split..]).unwrap();
        assert_eq!(
            events,
            [SseEvent {
                event: None,
                data: "ol\u{e1}".into()
            }]
        );
    }

    #[test]
    fn follows_the_event_stream_format() {
        let input = b": comment\r\nid: 7\r\nretry: 10\r\nevent: a\r\ndata: one\r\ndata:two\r\n\r\n\
                      event: no-data\n\n\
                      data\rdata: \r\r\
                      data: tail";
        let (events, parser) = parse_in_pieces(input, 3);
        assert_eq!(
            events,
            [
                SseEvent {
                    event: Some("a".into()),
                    data: "one\ntwo".into()
                },
                SseEvent {
                    event: None,
                    data: "\n".into()
                },
            ]
        );
        assert!(parser.has_pending(), "the last event was never closed");
    }

    #[test]
    fn rejects_broken_streams() {
        let mut parser = SseParser::new();
        assert!(matches!(
            parser.push(b"data: \xff\n"),
            Err(ProviderError::Malformed(_))
        ));
        let mut parser = SseParser::new();
        let huge = vec![b'a'; MAX_LINE_BYTES + 1];
        assert!(matches!(
            parser.push(&huge),
            Err(ProviderError::Malformed(_))
        ));
    }
}
