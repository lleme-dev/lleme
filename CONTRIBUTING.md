# Contributing to Lleme

Thank you for your interest in Lleme. This document explains how to
propose a change and what is expected of a pull request.

Related documents:

- `ARCHITECTURE.md`: structure of the code and naming conventions
- `GIT_CONVENTIONS.md`: branches, commit messages and versions
- `AI_POLICY.md`: rules for using AI tools
- `CLA.md`: contributor license agreement (`CLA.pt-BR.md` in Portuguese)
- `SECURITY.md`: how to report a vulnerability
- `CODE_OF_CONDUCT.md`: how we treat each other

## Ways to contribute

- Report a bug, with the steps to reproduce it.
- Propose a feature or a design change in an issue.
- Improve the documentation.
- Send a fix or a feature in a pull request.

Do not report security problems in a public issue. Follow
`SECURITY.md`.

## Before you write code

Small fixes can go straight to a pull request.

A larger change starts with an issue: a new feature, any change to the
public API, to the data model or to a port in the core crate, or more
than about 400 changed lines of production code (see "Pull requests"
for what counts). Describe the problem and the design you have in
mind, and wait for a maintainer to agree before you build it. This
avoids work that cannot be merged.

## Project layout

```
crates/core/     domain, use cases, ports, provider format translation
crates/server/   native binary
crates/edge/     Cloudflare Workers build
portal/          web portal
```

The core crate must compile to `wasm32-unknown-unknown` and must not
depend on a runtime. `ARCHITECTURE.md` has the dependency rules.

## Development setup

You need the Rust toolchain pinned in `rust-toolchain.toml`, which
`rustup` installs on first use, and Node.js with pnpm for the portal.

Checks that CI runs, and that should pass before you open a pull
request:

```
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check -p lleme-core --target wasm32-unknown-unknown
cargo deny check
```

The portal lives in `portal/` and uses pnpm. Its commands will be
listed here when the portal code lands.

CI uses path filters: a change only under `portal/` does not run the
Rust jobs, and the reverse.

## Pull requests

1. Create a branch from `main`, named as described in
   `GIT_CONVENTIONS.md`, for example `feat/12-context-filter`.
2. Open the pull request as a draft early, so CI runs.
3. Fill in the pull request template. The "AI usage" section is
   required for contributors; see `AI_POLICY.md`.
4. Review your own diff in the GitHub interface, then tick "I reviewed
   the full diff myself". The `pr-check` job fails while that box is
   unticked.
5. Mark the pull request as ready for review.

What we look for:

- **One purpose per pull request.** A preparatory refactor goes in its
  own pull request, before the behavior change.
- **Reviewable in about 30 minutes.** That means about 400 changed
  lines of production code at most. Production code is what lives
  under `crates/*/src` and `portal/src`, excluding tests.
- **Not counted toward the 400:** tests (`tests/` directories,
  `#[cfg(test)]` modules, `*.test.*` and `*.spec.*` files), fixtures,
  snapshots, SQL migrations, `Cargo.lock`, `pnpm-lock.yaml` and
  generated code such as the OpenAPI client.
- **A hard ceiling of about 1,000 changed lines in total**, everything
  included. Above it, split the pull request even if production code
  is under 400.
- **Tests for the new behavior**, and for the bug when it is a fix,
  in the same pull request as the code they cover. No follow-up pull
  request that only adds tests.
- **Documentation updated** when behavior or configuration changes.
- **A title in Conventional Commits format.** Pull requests are merged
  by squash, and the title becomes the commit message.

Commits must be signed. `GIT_CONVENTIONS.md` explains the format of
commit messages.

## Code rules

- Every source file starts with the license identifier:
  `// SPDX-License-Identifier: AGPL-3.0-only`
- No `unwrap` or `expect` outside tests and `main`.
- Follow the naming conventions in `ARCHITECTURE.md`. Data is
  `snake_case` end to end.
- The portal talks to the gateway only through the documented public
  API.

## Dependencies

New dependencies need a reason in the pull request description. Their
licenses must be on the allowed list in `deny.toml`: MIT, Apache-2.0,
BSD, ISC, MPL-2.0 and Unicode. CI rejects anything else.

If you include or adapt code from another project, say so in the pull
request, with the source and its license.

## Contributor license agreement

Lleme is licensed under the GNU Affero General Public License,
version 3. To have a contribution merged you accept the contributor
license agreement in `CLA.md`. It is a license, not a transfer: you
keep ownership of what you write.

On your first pull request a bot asks you to accept it. You do so once,
by posting this comment on the pull request:

> I have read the CLA Document and I hereby sign the CLA

Minor contributions, such as a spelling or formatting fix, or a small
documentation correction with no code, may be accepted without it. A
maintainer decides case by case.

If you contribute as part of your job, check that your employer allows
it. Section 7 of the agreement covers contributions on behalf of a
company.

## AI usage

AI tools are allowed, and their use must be disclosed in the pull
request. Read `AI_POLICY.md` before your first contribution.

The maintainers use AI tools in their own work. This statement is
their standing disclosure, so their pull requests do not repeat it one
by one. They follow the same standard as everyone else: a human reads,
tests and answers for every change that is merged.

## Review

A maintainer reviews every pull request. Automated review may leave
comments first; they are suggestions, and a maintainer makes the
decision. Please reply to review comments in your own words.

The project has one maintainer at the moment, so a first response can
take a few days.

## Conduct

Everyone taking part in the project is expected to follow
`CODE_OF_CONDUCT.md`.
