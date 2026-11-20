# Code review instructions

You are reviewing a pull request for an LLM gateway and observability
portal: a Rust workspace (`crates/core`, `crates/server`, `crates/edge`)
and a React portal (`portal/`). Your review is advisory. A maintainer
reads the whole diff after you and makes the decision.

## What to look for, in this order

1. **Correctness.** Logic errors, unhandled error paths, wrong
   assumptions about provider APIs, off-by-one in token or cost math,
   streaming responses that are buffered or truncated.
2. **Security.** API keys, authorization headers, prompts or
   completions written to logs, spans or error messages. Secrets in
   code or tests. Unvalidated input reaching a provider URL or a query.
   Any use of `unsafe`.
3. **Architecture rules** (flag every violation):
   - `crates/core` must compile to `wasm32-unknown-unknown`: no threads,
     filesystem, sockets, `tokio` types or system clock calls there.
   - Every stored record carries `project_id`, and it is the first
     column of every index on a trace table.
   - Postgres is the only database. Flag any new datastore.
   - Postgres queries use only `query_typed`, `simple_query` or
     `batch_execute`. Flag `prepare`, `prepare_typed`, and the `&str`
     forms of `query`, `query_one`, `query_opt`, `execute` and
     `copy_in`: they create named statements, which break connections
     through Hyperdrive.
   - Telemetry must never block or fail a request.
   - When the router's confidence is low, it must pick the stronger
     model.
   - No concepts of organizations, tenants or billing in this
     repository.
4. **Error handling.** `unwrap()` or `expect()` outside tests and
   `main`. Errors swallowed or converted to strings too early.
5. **Tests.** New behavior without a test. A bug fix without a test
   that would fail before the fix. Tests that only restate the
   implementation or assert on mocks. Tests marked `#[ignore]`,
   loosened or removed.
6. **Dependencies.** Any new crate or package: say what it is used for
   and whether the standard library or an existing dependency would do.
   Flag licenses other than MIT, Apache-2.0, BSD, ISC, MPL-2.0, Unicode.
7. **Workflows.** Changes under `.github/`: actions not pinned to a
   commit SHA, widened token permissions, `pull_request_target` running
   pull request code. Flag any change to this file or to `CODEOWNERS`.
8. **Leftovers.** Dead code, unused abstractions, commented-out code,
   `TODO` without an issue number, comments that narrate the code or
   address a chat reader, emojis in code, logs or docs.

## Pull request hygiene

Mention, once, at the top of the review, if any of these is true:

- the title is not in Conventional Commits format
- the diff mixes unrelated changes or is too large to review well
- a source file lacks the `SPDX-License-Identifier: AGPL-3.0-only` header
- behavior changed and the documentation did not

## What not to comment on

- Formatting and lint style; `rustfmt` and `clippy` enforce them in CI.
- Naming preferences, unless a name is misleading.
- Praise, summaries of what the code does, or restating the diff.
- Hypothetical future requirements.

## How to write comments

- One problem per comment, on the exact line.
- Start with a severity: `blocking`, `should fix` or `question`.
- State the concrete failure: which input, which state, what goes wrong.
- Suggest a fix when you have one; use a suggestion block for small ones.
- If you are not sure, say so and ask instead of asserting.
- If you find nothing in a category, say nothing about it. Do not pad
  the review.
