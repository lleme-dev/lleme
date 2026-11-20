# Architecture

Lleme is an LLM gateway with routing and observability. It runs as a
native server in a container and as a Cloudflare Worker. This document
describes how the code is organized and the conventions it follows.

## Ports and adapters

The project uses a hexagonal (ports and adapters) architecture. The
reason is practical: the same logic runs on two runtimes, so the core
has to exist without any I/O.

```
 inbound adapters          core                 outbound adapters
+------------------+   +-----------------+   +--------------------+
| OpenAI/Anthropic |-->| domain          |-->| provider HTTP      |
| compatible API   |   | use cases       |   +--------------------+
+------------------+   | ports (traits)  |   +--------------------+
| admin API        |-->| provider format |-->| Postgres:          |
+------------------+   | translation     |   | config and traces  |
| OTLP ingest      |-->|                 |   |                    |
+------------------+   +-----------------+   +--------------------+

 crates/server: Tokio + Hyper        crates/edge: workers-rs (WASM)
 both implement the same ports
```

The core defines ports as traits. Each runtime brings its own adapters
for those ports. Nothing in the core knows which runtime it is in.

### Crates

| Crate          | Contains                                              | May depend on            |
| -------------- | ----------------------------------------------------- | ------------------------ |
| `lleme-core`   | Domain types, use cases, ports, provider translation  | I/O-free libraries only  |
| `lleme-server` | Native adapters, inbound HTTP, composition, `main`    | `lleme-core`             |
| `lleme-edge`   | Workers adapters, composition                         | `lleme-core`             |

Dependency rules, enforced in CI:

- `core` never depends on `server` or `edge`.
- `server` and `edge` never depend on each other.
- `core` builds for `wasm32-unknown-unknown` on every pull request.
- In `core`, these are denied by lint: `tokio`, `std::fs`, `std::net`,
  `std::thread`, and direct reads of the system clock.

### Inside the core

```
crates/core/src/
  domain/      pure types and rules: Request, Span, Policy, Money
  ports/       traits: HttpClient, KeyStore, SpanSink, Clock
  app/         use cases: proxy, route, record
  providers/   OpenAI, Anthropic, Gemini <-> internal format
```

Provider adapters live in the core because they are pure format
translation. The network call itself goes through the `HttpClient`
port.

### Writing ports for two runtimes

Tokio needs `Send + Sync` to move work across threads. WebAssembly is
single-threaded, and the JavaScript types a Worker handles are not
`Send`. Ports must compile for both, so:

- Bounds use `MaybeSend` and `MaybeSync`, defined in `core`: they mean
  `Send` and `Sync` on native targets and nothing on `wasm32`.
- Async methods return `impl Future<Output = T> + MaybeSend`. The
  `async_trait` macro is not used.
- Nothing in `core` calls `Instant::now()` or `SystemTime::now()`; it
  panics on `wasm32`. Time comes from the `Clock` port.

### Database access

Both runtimes use `tokio-postgres`, without an ORM or `sqlx`. The SQL
and the row mapping are shared; only how the connection is opened
differs. The server keeps a pool. The Worker opens one connection per
request to Hyperdrive, which pools behind it, and keeps keys and
policies in a short-lived cache so the database stays off the request
path.

Queries use only `query_typed`, which sends an unnamed statement with
its parameter types, and `simple_query` or `batch_execute`. Named
prepared statements are not used: through Hyperdrive, the `Close`
message that `tokio-postgres` sends when a named statement is dropped
closes the connection, and the next query on it fails. The `&str`
forms of `query`, `query_one`, `query_opt`, `execute` and `copy_in`
prepare and drop a named statement on every call, so they are out too.
The same rule holds on the server, so a query is written one way and
tested once. The local Workers emulator connects to Postgres directly,
so no test there would catch a violation.

### Trace storage

Postgres is the only database. It holds configuration and traces on
both runtimes.

The core sees trace storage through two ports: `SpanSink` for writes
and `TraceStore` for reads. Both have a Postgres adapter. The server
batches spans in memory and writes them with multi-row inserts. The
Worker writes its batch after the response is sent.

Trace tables are append-only and partitioned by month. Retention drops
whole partitions. Prompt and completion content lives in a separate
table with its own, shorter retention. Dashboards read rollup tables,
not raw spans.

The ports exist so that another store can be added later without
touching the core: a columnar database for high volume, or object
storage for archives. Neither is part of the first version.

### Request path

1. An inbound adapter parses the request into the normalized type.
2. `app::route` filters candidate models, applies the active policy,
   and records a route decision.
3. `app::proxy` translates the request for the chosen provider and
   sends it through `HttpClient`, streaming the response back.
4. `app::record` builds a span and hands it to `SpanSink`. This never
   blocks or fails the request.

## Code patterns

| Topic         | Pattern                                                                 |
| ------------- | ----------------------------------------------------------------------- |
| Ports         | Traits named for the capability: `KeyStore`, `SpanSink`, `Clock`        |
| Adapters      | `<Technology><Port>`: `PostgresKeyStore`, `PostgresSpanSink`            |
| Composition   | Wired in one place per runtime; no global state, no singletons          |
| Configuration | Read and validated once at startup into a typed struct                  |
| Errors        | Typed enum per crate with `thiserror`; `anyhow` only in `main`          |
| Identifiers   | One type per id: `ProjectId`, `KeyId`, `TraceId`                        |
| Money         | Integer nano-dollars; never floating point                              |
| Time          | Always UTC; read through the `Clock` port                               |
| Database      | Explicit SQL, no ORM; forward-only migrations                           |
| Tests         | Core with port fakes; adapters against real databases; few end-to-end   |
| Logs          | Structured JSON; never a prompt, completion or key                      |
| Language      | Code, comments, commits and documentation in English                    |

Rules that follow from the design:

- Every stored record carries `project_id`, and it is the first
  column of every index on a trace table.
- Telemetry is fire-and-forget with a bounded buffer.
- When the router is unsure, it picks the stronger model.
- Cost is computed at the gateway from provider-reported usage and the
  price in force on that date.

## Naming conventions

Data is `snake_case` everywhere: columns, JSON fields, query parameters
and metrics. A field keeps the same name from the database to the API.
Rust and Postgres already use it, and so do the OpenAI and
Anthropic APIs that the gateway mirrors.

| Where                              | Convention                         | Example                                   |
| ---------------------------------- | ---------------------------------- | ----------------------------------------- |
| Rust functions, variables, modules | `snake_case`                       | `route_request`, `input_tokens`           |
| Rust types, traits, enum variants  | `UpperCamelCase`                   | `RouteDecision`, `SpanSink`               |
| Rust constants and statics         | `SCREAMING_SNAKE_CASE`             | `DEFAULT_TIMEOUT_MS`                      |
| Acronyms in Rust names             | As a word                          | `HttpClient`, `LlmRequest`                |
| Crates                             | `kebab-case` with prefix           | `lleme-core`                              |
| Tables                             | `snake_case`, plural               | `virtual_keys`, `route_decisions`         |
| Columns                            | `snake_case`                       | `input_tokens`, `policy_version`          |
| Primary key                        | `id`                               | `id uuid`                                 |
| Foreign key                        | singular + `_id`                   | `project_id`, `policy_id`                 |
| Timestamps                         | `_at` suffix, with time zone       | `created_at timestamptz`                  |
| Booleans                           | `is_` or `has_` prefix             | `is_shadow`, `has_tools`                  |
| Ambiguous units                    | Unit suffix                        | `duration_ms`, `cost_nano_usd`            |
| Indexes and constraints            | Prefix by kind                     | `idx_spans_project_start`, `uq_virtual_keys_hash` |
| API JSON fields                    | `snake_case`                       | `"input_tokens"`, `"created_at"`          |
| HTTP paths                         | `kebab-case`, plural, versioned    | `/api/v1/virtual-keys`                    |
| Query parameters                   | `snake_case`                       | `?start_time=...&page_size=50`            |
| Custom headers                     | Project prefix, no `X-`            | `Lleme-Trace-Id`                          |
| Environment variables              | `SCREAMING_SNAKE_CASE` with prefix | `LLEME_DATABASE_URL`                      |
| Prometheus metrics                 | `snake_case` with unit and suffix  | `lleme_spans_dropped_total`               |
| Own telemetry attributes           | Lowercase, dotted                  | `lleme.route.policy_version`              |
| TypeScript variables and functions | `camelCase`                        | `fetchTraces`                             |
| TypeScript components and types    | `PascalCase`                       | `TraceList`                               |
| TypeScript files                   | `kebab-case`                       | `trace-list.tsx`                          |

Points that often cause doubt:

- **Proxy endpoints do not follow these conventions.** They copy the
  OpenAI and Anthropic formats field for field. The conventions apply
  to the admin API.
- **In the portal, API data stays in `snake_case`.** Types are
  generated from the OpenAPI document and used as they come. Only
  local TypeScript variables are `camelCase`.
- **In trace tables, column names come from the internal schema**, not
  from dotted OpenTelemetry attributes. `gen_ai.usage.input_tokens` is
  stored as `input_tokens`; the mapping layer does the translation.
- **Plural table, singular foreign key.** The table is `policies`; the
  column that points to it is `policy_id`.

## Types and formats

| Data         | In the database                                      | In the API                            |
| ------------ | ---------------------------------------------------- | ------------------------------------- |
| Identifiers  | `uuid`, generated as UUIDv7                          | String                                |
| Money        | Integer, nano-dollars                                | Integer, same `cost_nano_usd` field   |
| Tokens       | Unsigned integer                                     | Integer                               |
| Duration     | Integer, milliseconds                                | Integer                               |
| Timestamps   | `timestamptz`, always UTC                            | RFC 3339 string in UTC                |
| Enumerations | Text with a check constraint                         | `snake_case` string                   |
| Virtual keys | Hash only                                            | Full value once, at creation, prefixed `lleme_sk_` |

## Admin API

- The OpenAPI document is the source of truth and is versioned in this
  repository. The portal client is generated from it.
- Everything lives under `/api/v1`. A breaking change creates `/api/v2`.
- Errors use Problem Details (RFC 9457) with a stable `snake_case`
  `code` field.
- Cursor pagination with `page_size` and `next_cursor`.
- List responses are `{"data": [...], "next_cursor": ...}`, never a
  bare array.
- The portal talks to the gateway only through this API.

## Deliberately left out

- **Full DDD layering** with aggregates, generic repositories and
  domain events. The domain is small.
- **An ORM.** There are few queries, and the trace tables need
  hand-written SQL anyway.
- **A second database.** A columnar store for traces is an optional
  adapter for later, not a requirement to install Lleme.
- **Microservices.** One process with roles (`all`, `gateway`,
  `ingest`, `api`).
