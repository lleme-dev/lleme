#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Records provider responses used as test fixtures by the core adapters.
# Prompts are synthetic. Response ids are rewritten to fixed values, and
# only status and body are kept: response headers carry account and
# request identifiers.
#
#   ANTHROPIC_API_KEY=... OPENAI_API_KEY=... OPENAI_MODEL=... scripts/record-fixtures.sh
#
# Optional: ANTHROPIC_MODEL (default claude-haiku-5-5), PROVIDERS (by
# default "anthropic openai anthropic_cache openai_cache"). Fourteen
# requests in all; the cache cases send a synthetic prompt of about 3000
# tokens twice, so the first writes the provider's prompt cache and the
# second reads it.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/crates/core/fixtures"
providers="${PROVIDERS:-anthropic openai anthropic_cache openai_cache}"
ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-haiku-5-5}"

PLAIN='Reply with exactly this sentence and nothing else: Hello from the fixture.'
TOOL_PROMPT='What is the weather in Lima? Use the tool.'
NO_MODEL='lleme-fixture-no-such-model'
LONG="$(for i in $(seq 1 150); do printf 'Fixture paragraph %s describes nothing in particular and exists only to make the prompt long enough to cache. ' "$i"; done)"

# Replaces each distinct id with a numbered placeholder, the same one for
# every occurrence in the file, so tests can still match ids across events.
scrub() {
  perl -0pi -e '
    my %seen; my $n = 0;
    s{\b(msg|toolu|req|chatcmpl|call|fp)([_-])[A-Za-z0-9]{6,}\b}{
      $seen{$&} //= sprintf("%s%sfixture%02d", $1, $2, ++$n); $seen{$&}
    }ge;
  ' "$1"
}

# post <provider> <file> <expected status> <json body>
post() {
  local provider="$1" file="$2" expected="$3" body="$4" status
  mkdir -p "$out/$provider"
  if [ "$provider" = anthropic ]; then
    status="$(curl -sS -o "$out/$provider/$file" -w '%{http_code}' https://api.anthropic.com/v1/messages \
      -H @<(printf 'x-api-key: %s\n' "$ANTHROPIC_API_KEY") \
      -H 'anthropic-version: 2026-10-21' -H 'content-type: application/json' --data-binary "$body")"
  else
    status="$(curl -sS -o "$out/$provider/$file" -w '%{http_code}' https://api.openai.com/v1/chat/completions \
      -H @<(printf 'authorization: Bearer %s\n' "$OPENAI_API_KEY") \
      -H 'content-type: application/json' --data-binary "$body")"
  fi
  scrub "$out/$provider/$file"
  if [ "$status" != "$expected" ]; then
    echo "$provider/$file: expected status $expected, got $status" >&2
    exit 1
  fi
  echo "$provider/$file: $status, $(wc -c < "$out/$provider/$file" | tr -d ' ') bytes"
}

anthropic() {
  : "${ANTHROPIC_API_KEY:?missing ANTHROPIC_API_KEY}"
  local tools='[{"name":"get_weather","description":"Current weather for a city.","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]'
  local m="$ANTHROPIC_MODEL"
  post anthropic chat.json 200 "$(jq -nc --arg m "$m" --arg p "$PLAIN" '{model:$m,max_tokens:50,messages:[{role:"user",content:$p}]}')"
  post anthropic chat-stream.sse 200 "$(jq -nc --arg m "$m" --arg p "$PLAIN" '{model:$m,max_tokens:50,stream:true,messages:[{role:"user",content:$p}]}')"
  post anthropic tool.json 200 "$(jq -nc --arg m "$m" --arg p "$TOOL_PROMPT" --argjson t "$tools" '{model:$m,max_tokens:200,tools:$t,messages:[{role:"user",content:$p}]}')"
  post anthropic tool-stream.sse 200 "$(jq -nc --arg m "$m" --arg p "$TOOL_PROMPT" --argjson t "$tools" '{model:$m,max_tokens:200,stream:true,tools:$t,messages:[{role:"user",content:$p}]}')"
  post anthropic error-404.json 404 "$(jq -nc --arg m "$NO_MODEL" --arg p "$PLAIN" '{model:$m,max_tokens:50,messages:[{role:"user",content:$p}]}')"
}

openai() {
  : "${OPENAI_API_KEY:?missing OPENAI_API_KEY}" "${OPENAI_MODEL:?missing OPENAI_MODEL}"
  local tools='[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
  # Recent OpenAI reasoning models refuse function tools on Chat
  # Completions unless reasoning_effort is "none".
  local m="$OPENAI_MODEL" so='{"include_usage":true}'
  post openai chat.json 200 "$(jq -nc --arg m "$m" --arg p "$PLAIN" '{model:$m,max_completion_tokens:50,messages:[{role:"user",content:$p}]}')"
  post openai chat-stream.sse 200 "$(jq -nc --arg m "$m" --arg p "$PLAIN" --argjson so "$so" '{model:$m,max_completion_tokens:50,stream:true,stream_options:$so,messages:[{role:"user",content:$p}]}')"
  post openai tool.json 200 "$(jq -nc --arg m "$m" --arg p "$TOOL_PROMPT" --argjson t "$tools" '{model:$m,max_completion_tokens:200,reasoning_effort:"none",tools:$t,messages:[{role:"user",content:$p}]}')"
  post openai tool-stream.sse 200 "$(jq -nc --arg m "$m" --arg p "$TOOL_PROMPT" --argjson t "$tools" --argjson so "$so" '{model:$m,max_completion_tokens:200,reasoning_effort:"none",stream:true,stream_options:$so,tools:$t,messages:[{role:"user",content:$p}]}')"
  post openai error-404.json 404 "$(jq -nc --arg m "$NO_MODEL" --arg p "$PLAIN" '{model:$m,max_completion_tokens:50,messages:[{role:"user",content:$p}]}')"
}

# A one-hour cache write, so the fixture shows how the long-lived cache is
# reported, then a read of the same prefix.
anthropic_cache() {
  : "${ANTHROPIC_API_KEY:?missing ANTHROPIC_API_KEY}"
  local body
  body="$(jq -nc --arg m "$ANTHROPIC_MODEL" --arg long "$LONG" --arg p "$PLAIN" \
    '{model:$m,max_tokens:50,system:[{type:"text",text:$long,cache_control:{type:"ephemeral",ttl:"1h"}}],messages:[{role:"user",content:$p}]}')"
  post anthropic cache-write.json 200 "$body"
  post anthropic cache-read.json 200 "$body"
}

# OpenAI caches long prefixes on its own; the request names no cache.
openai_cache() {
  : "${OPENAI_API_KEY:?missing OPENAI_API_KEY}" "${OPENAI_MODEL:?missing OPENAI_MODEL}"
  local body
  body="$(jq -nc --arg m "$OPENAI_MODEL" --arg long "$LONG" --arg p "$PLAIN" \
    '{model:$m,max_completion_tokens:50,messages:[{role:"developer",content:$long},{role:"user",content:$p}]}')"
  post openai cache-write.json 200 "$body"
  post openai cache-read.json 200 "$body"
}

for p in $providers; do "$p"; done

# A key must never land in a fixture.
if grep -rIl -E 'sk-[A-Za-z0-9_-]{10,}|sk-ant-' "$out" 2>/dev/null; then
  echo "a fixture above contains something that looks like an API key; delete it" >&2
  exit 1
fi
