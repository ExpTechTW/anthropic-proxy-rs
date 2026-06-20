# anthropic-proxy-rs

**English** · [繁體中文](README.zh-TW.md)

[![CI & Release](https://github.com/ExpTechTW/anthropic-proxy-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ExpTechTW/anthropic-proxy-rs/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/ExpTechTW/anthropic-proxy-rs?sort=date)](https://github.com/ExpTechTW/anthropic-proxy-rs/releases)

High-performance Rust proxy that translates the **Anthropic Messages API** into the **OpenAI Chat Completions** format. Point Claude Code, Claude Desktop, or any Anthropic API client at OpenRouter, native OpenAI, Azure OpenAI, or any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, a private gateway, …).

> Fork of [m0n0x41d/anthropic-proxy-rs](https://github.com/m0n0x41d/anthropic-proxy-rs) maintained by [ExpTech](https://github.com/ExpTechTW). Adds `tool_choice` / `metadata` / `refusal` support, a `count_tokens` endpoint, upstream status-code preservation, and connection-stability hardening.

## Features

- **Fast & lightweight** — Rust with async I/O (~3 MB binary)
- **Full streaming** — Server-Sent Events (SSE) with real-time responses
- **Tool calling** — function/tool calling incl. `tool_choice`
- **Universal** — any OpenAI-compatible API (OpenRouter, OpenAI, Azure, local LLMs)
- **Extended thinking** — detects Claude's reasoning mode and routes models accordingly
- **Web search & fetch** — emulates Anthropic's server-side `web_search` / `web_fetch` for models that can't browse, via a bundled [open-websearch](https://github.com/aas-ee/open-websearch)
- **Self-learning skills** *(optional, off by default)* — learns reusable lessons from the conversations it serves, web-verifies them through a trust gate, and injects relevant ones into future requests (Qdrant-backed, no fine-tuning). See [Self-learning skills](#self-learning-skills)
- **Resilient** — retries transient failures, preserves upstream status codes, 600 s request timeout
- **Drop-in** — works with the official Anthropic SDKs and Claude Code

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/messages` | Main completion endpoint (streaming + non-streaming). Also accepts a trailing slash. |
| `POST` | `/v1/messages/count_tokens` | Token count — exact via the upstream `/tokenize` when enabled, otherwise a local BPE estimate |
| `GET`  | `/v1/models` | Lists models reported by the upstream, translated to Anthropic shape |
| `GET`  | `/health` | Liveness check (`OK`) |
| `GET`  | `/metrics` | Prometheus metrics |

## Download

Prebuilt binaries are published to [GitHub Releases](https://github.com/ExpTechTW/anthropic-proxy-rs/releases) for:

| Platform | Asset |
|----------|-------|
| Linux x86_64 (static musl) | `anthropic-proxy-x86_64-unknown-linux-musl.tar.gz` |
| Linux arm64 (static musl) | `anthropic-proxy-aarch64-unknown-linux-musl.tar.gz` |
| macOS arm64 (Apple Silicon) | `anthropic-proxy-aarch64-apple-darwin.tar.gz` |

```bash
# Linux x86_64 — latest full release
curl -fsSL https://github.com/ExpTechTW/anthropic-proxy-rs/releases/latest/download/anthropic-proxy-x86_64-unknown-linux-musl.tar.gz \
  | tar -xz && ./anthropic-proxy --version
```

Each asset has a matching `.sha256` for verification. Linux binaries are statically linked (musl) and run on any distribution.

**Release channels & versioning** — versions are date-based: `vYYYY.MM.DD+build.N`, where `N` resets to `1` each UTC day.

- Pushes to **`main`** publish a **pre-release** (`… Pre-release (auto)`).
- Pushes to **`release`** publish a **full release** (`… Release (auto)`) — this is what `releases/latest` resolves to.

## Quick Start

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build and run

```bash
cargo build --release
UPSTREAM_BASE_URL=https://api.openai.com \
UPSTREAM_API_KEY=sk-... \
./target/release/anthropic-proxy
```

### Install with Cargo

```bash
cargo install --git https://github.com/ExpTechTW/anthropic-proxy-rs --locked
```

### Run from anywhere

```bash
UPSTREAM_BASE_URL=https://openrouter.ai/api \
UPSTREAM_API_KEY=sk-or-... \
anthropic-proxy
```

### Docker

The repository's [`Dockerfile`](Dockerfile) **downloads a prebuilt release binary** — no Rust toolchain, fast builds. Choose the channel with build args:

```bash
# Latest full release (default)
docker build -t anthropic-proxy .

# Latest pre-release (main-branch builds)
docker build -t anthropic-proxy --build-arg CHANNEL=prerelease .

# Pin an exact tag (overrides CHANNEL; works for pre-releases too)
docker build -t anthropic-proxy --build-arg VERSION=v2026.06.05+build.3 .

# Multi-arch
docker buildx build --platform linux/amd64,linux/arm64 -t anthropic-proxy .

docker run -p 3000:3000 \
  -e UPSTREAM_BASE_URL=https://openrouter.ai/api \
  -e UPSTREAM_API_KEY=sk-or-... \
  anthropic-proxy
```

| Build arg | Default | Meaning |
|-----------|---------|---------|
| `CHANNEL` | `release` | `release` = latest full release · `prerelease` = latest pre-release |
| `VERSION` | (empty) | Pin an exact tag, e.g. `v2026.06.05+build.3` (overrides `CHANNEL`) |

To compile from source instead, use [`Dockerfile.source`](Dockerfile.source):

```bash
docker build -f Dockerfile.source -t anthropic-proxy .
```

> Both images bundle Node + [open-websearch](https://github.com/aas-ee/open-websearch) (started alongside the proxy) plus an optional SSH egress-proxy pool, powering the `web_search` / `web_fetch` emulation — see [Web search & fetch](#web-search--fetch).

## Use with Claude Code

```bash
anthropic-proxy --daemon && ANTHROPIC_BASE_URL=http://localhost:3000 claude
```

### Recommended settings — [`examples/claude-code-settings.json`](examples/claude-code-settings.json)

Copy it into `~/.claude/settings.json` (or a project-level `.claude/settings.json`) and fill in your proxy URL and key. The one setting that really matters:

**`CLAUDE_CODE_AUTO_COMPACT_WINDOW`** — set it to your upstream model's **real** context window in tokens (e.g. `105120`). When a smaller backend is mapped to a name like `claude-sonnet-4-5`, Claude Code otherwise assumes that model's full 200K/1M window and won't auto-compact until *far* past the real limit — so requests eventually fail with `context length exceeded`.

> ⚠️ **Not `CLAUDE_CODE_MAX_CONTEXT_TOKENS`** — per the [docs](https://code.claude.com/docs/en/env-vars) it only applies together with `DISABLE_COMPACT`. `CLAUDE_CODE_AUTO_COMPACT_WINDOW` is the one that drives auto-compaction.
>
> **No `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` needed** — Claude Code reserves a fixed ~33K-token buffer, so a smaller window already compacts with headroom (~68% of 105K). That override only compacts *earlier* (it's ignored above the default), so it can't delay compaction.

Verify with `/context`: it should show your real window (e.g. `30.6k / 105.1k`) and an `Auto-compact window` line.

### Status line — [`examples/statusline.sh`](examples/statusline.sh)

A companion status line showing the **real** context fill (matching `/context`, not Claude Code's full-window `used_percentage`) and a running session cost:

```text
Sonnet 4.5 │ ctx ━━━━━┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ 29% 105k │ NT$6.38
```

The cost is summed from the session transcript, so it follows the session — delete the session and it's gone, no orphaned state file. Install (needs `jq`), then set your gateway's rates near the top of the script:

```bash
cp examples/statusline.sh ~/.claude/statusline.sh && chmod +x ~/.claude/statusline.sh
```

## Configuration

### Command-line options

```bash
anthropic-proxy --help
```

**Commands**

| Command | Description |
|---------|-------------|
| `stop` | Stop a running daemon |
| `status` | Check daemon status |

**Options**

| Option | Short | Description |
|--------|-------|-------------|
| `--config <FILE>` | `-c` | Path to a custom `.env` file |
| `--debug` | `-d` | Enable debug logging |
| `--verbose` | `-v` | Verbose logging (logs full request/response bodies) |
| `--port <PORT>` | `-p` | Port to listen on (overrides `PORT`) |
| `--bind <ADDR>` | | Listener bind address (overrides `ANTHROPIC_PROXY_BIND`, default `0.0.0.0`) |
| `--system-prompt-ignore <TEXT>` | | Remove system-prompt terms before forwarding (repeat or separate with `;`) |
| `--daemon` | | Run as a background daemon |
| `--pid-file <FILE>` | | PID file path (default `/tmp/anthropic-proxy.pid`) |
| `--help` | `-h` | Print help |
| `--version` | `-V` | Print version |

### Environment variables

Set via environment or a `.env` file:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `UPSTREAM_BASE_URL` | **Yes** | – | OpenAI-compatible endpoint URL |
| `UPSTREAM_API_KEY` | No\* | – | API key for the upstream service |
| `UPSTREAM_API_KEY_PASSTHROUGH` | No | `false` | Take the API key from each request's `x-api-key` header (`true`/`false`) |
| `PORT` | No | `3000` | Server port |
| `ANTHROPIC_PROXY_BIND` | No | `0.0.0.0` | Bind address. Use `127.0.0.1` to restrict to localhost. Binding to `0.0.0.0` logs a warning. |
| `ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS` | No | – | System-prompt terms to remove before forwarding (`;` or newline separated) |
| `ANTHROPIC_PROXY_MODEL_MAP` | No | – | Exact model remapping before the upstream call (`source=target;other=target`) |
| `ANTHROPIC_PROXY_UPSTREAM_TOKENIZE` | No | `false` | Use the upstream's vLLM-style `/tokenize` for **exact** `count_tokens` and accurate overflow clamping, instead of a local estimate |
| `ANTHROPIC_PROXY_EFFORT_MAP` | No | – | Map Anthropic thinking → the OpenAI `reasoning_effort` field (see below) |
| `ANTHROPIC_PROXY_WEBSEARCH_MODEL` | No | – | Model the `web_search`/`web_fetch` agent loop routes through (e.g. `auto`). **Unset disables emulation** (the web tools are stripped). See [Web search & fetch](#web-search--fetch) |
| `ANTHROPIC_PROXY_WEBSEARCH_URL` | No | `http://localhost:3100/mcp` | open-websearch MCP endpoint used for the emulation |
| `ANTHROPIC_PROXY_HEARTBEAT_SECS` | No | `15` | Seconds of streaming-output silence before an SSE `: keep-alive` comment (stops a fronting proxy timing out before the first token / during a search). `0` disables |
| `REASONING_MODEL` | No | (request model) | Model used when extended thinking is enabled\*\* |
| `COMPLETION_MODEL` | No | (request model) | Model used for standard requests (no thinking)\*\* |
| `DEBUG` | No | `false` | Debug logging (`1` or `true`) |
| `VERBOSE` | No | `false` | Verbose logging (`1` or `true`) |

\* Required if your upstream endpoint needs authentication.
\*\* The proxy detects when a request enables extended thinking (the `thinking` parameter) and routes it to `REASONING_MODEL`; requests without thinking use `COMPLETION_MODEL`. If neither is set, the model from the client request is used. `ANTHROPIC_PROXY_MODEL_MAP` is applied **after** this selection.

`UPSTREAM_BASE_URL` accepts any of these forms:
- Service base URL: `https://api.openai.com` → `/v1/chat/completions`
- Versioned base URL: `https://gateway.company.internal/v2` → `/v2/chat/completions`
- Full endpoint: `https://gateway.company.internal/v2/chat/completions` → used as-is

### Reasoning effort

`ANTHROPIC_PROXY_EFFORT_MAP` forwards the OpenAI `reasoning_effort` field to the upstream (a compatible gateway resolves the level to a per-model thinking budget), derived from the client's thinking request — global tiers plus optional per-(client-)model overrides, each tier `effort:maxBudget`:

```bash
ANTHROPIC_PROXY_EFFORT_MAP="low:2048,medium:8192,high:16384;claude-haiku-3-5=low:512,high:16384"
```

- The client's `thinking.budget_tokens` (clamped to **16384**) selects the first tier whose `maxBudget ≥ budget`; above them all → the highest tier.
- A `;`-segment **without** `=` is the global default; `model=tiers` overrides one client model.
- A direct Anthropic `effort` field is forwarded as-is. The level is passed through verbatim — validating it and defaulting unknown/empty levels is the upstream's job (a compatible gateway defaults them to `medium`).
- List only tiers your upstream accepts (e.g. qwen takes `low`/`medium`/`high`; `xhigh`/`max` would `400` if unsupported). When unset, no `effort` is sent.

### Configuration file locations

The proxy searches for a `.env` file in this order:

1. Custom path from `--config`
2. Current working directory (`./.env`)
3. Home directory (`~/.anthropic-proxy.env`)
4. System-wide (`/etc/anthropic-proxy/.env`)

If none is found, environment variables from your shell are used.

### API key passthrough

With `UPSTREAM_API_KEY_PASSTHROUGH=true`, the proxy reads each request's `x-api-key` header (the standard Anthropic-client header) and forwards it as `Authorization: Bearer {key}` upstream — so every client authenticates with its own key instead of one static `UPSTREAM_API_KEY`.

```bash
# Passthrough mode (UPSTREAM_API_KEY must NOT be set)
UPSTREAM_API_KEY_PASSTHROUGH=true \
UPSTREAM_BASE_URL=https://openrouter.ai/api \
anthropic-proxy
```

Constraints:
- Cannot be combined with `UPSTREAM_API_KEY` — if both are set the proxy refuses to start.
- If passthrough is on and a request has no (or an empty) `x-api-key`, no `Authorization` header is sent upstream; the upstream decides whether to accept it.

### Model mapping

```bash
ANTHROPIC_PROXY_MODEL_MAP='claude-opus-4-7=openai/gpt-4.1;claude-haiku-3-5=openai/gpt-4.1-mini'
```

### System-prompt sanitization

Remove configured terms from `system` prompts before forwarding (useful when a gateway/WAF blocks certain phrases):

```bash
ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS='rm -rf;git reset --hard' anthropic-proxy
# or
anthropic-proxy --system-prompt-ignore 'rm -rf' --system-prompt-ignore 'git reset --hard'
```

### Running as a daemon

```bash
anthropic-proxy --daemon            # start
anthropic-proxy status              # check
anthropic-proxy stop                # stop
tail -f /tmp/anthropic-proxy.log    # logs
```

## Web search & fetch

The proxy can **emulate Anthropic's server-side `web_search` and `web_fetch` tools** so models that can't browse (local LLMs, most OpenAI-compatible backends) still answer with live web data. When a request carries a `web_search_*` / `web_fetch_*` server tool, the proxy:

1. rewrites it into a callable function tool the backend model can invoke,
2. runs the tool loop itself — calling a bundled [**open-websearch**](https://github.com/aas-ee/open-websearch) (DuckDuckGo) for each search/fetch and feeding the results back to the model until it answers (bounded by a per-request search/fetch budget),
3. returns faithful Anthropic content blocks — `server_tool_use` + `web_search_tool_result` / `web_fetch_tool_result` + the answer text — for both streaming and non-streaming.

Long, multi-round searches are kept alive with SSE `: keep-alive` heartbeats (see `ANTHROPIC_PROXY_HEARTBEAT_SECS`) so a fronting proxy (e.g. Cloudflare's 100 s idle limit) doesn't drop the stream. The agent loop forces low effort and disables chain-of-thought for its own rounds to stay responsive.

Enable it by setting **`ANTHROPIC_PROXY_WEBSEARCH_MODEL`** to the model the loop should route through (e.g. `auto` to let a gateway load-balance across backends). When unset, emulation is **off** and the web tools are stripped from the request.

The provided [`Dockerfile`](Dockerfile) / [`Dockerfile.source`](Dockerfile.source) bundle Node + open-websearch and start it alongside the proxy (MCP on `:3100`), so no extra service is needed.

### Egress-proxy rotation (optional)

To spread searches across multiple source IPs and avoid single-IP rate-limiting, mount a proxy-list file at `/etc/websearch-ssh-proxies.txt` — one `user@host[:port] password` per line (`#` comments allowed). The container opens an SSH SOCKS tunnel per host (auto-reconnecting) and fronts them with [glider](https://github.com/nadoo/glider) (round-robin + health checks) as a single HTTP proxy that open-websearch routes through. With no file mounted, searches go out directly. Keep this file outside the build context and out of version control — it holds credentials.

## Self-learning skills

An optional, default-off layer that lets the proxy **learn reusable lessons from the conversations it serves and inject relevant learned knowledge into future requests** — transparently to the client, and best-effort so a failure can never break a proxied request. Inspired by skill-library / experiential-learning agents (Voyager, ReasoningBank, ExpeL).

Knowledge lives in an external **[Qdrant](https://qdrant.tech)** vector store (no model fine-tuning), and every entry carries a **trust tier** — only `verified`/`trusted` ones are ever injected. The pipeline:

1. **Inject** *(on the request path)* — embed the user's latest message, retrieve the top-k most relevant `verified`/`trusted` skills, and append them as a system block. Injected skill ids are surfaced in an `x-injected-skills` response header. Injection is **on by default** when the feature is enabled; a client can opt a single request out with the request header **`x-skills-inject: off`** (also `false`/`0`/`no`).
2. **Distil** *(background)* — after a conversation, an LLM judge labels the outcome (success/failure, strict: absence of a complaint is **not** success) and extracts ≤3 general, reusable lessons (learning from both success and failure), written as `candidate` — never injectable.
3. **Verify** *(background loop)* — each candidate is corroborated against the open web by a *quarantined* reader (treats results as untrusted data, no tools); promotion to `verified` requires a positive verdict **and** independent multi-source corroboration (not the model's own confidence), then to `trusted` after a soak period.
4. **Curate** *(background loop)* — drop unverified candidates past the retention window and collapse near-duplicate entries, keeping the store small and high-signal.
5. **Proactive** *(background loop, opt-in)* — record the questions users actually ask, then research recent ones on the web and distil candidates, so the store grows toward what's needed.

Everything is non-parametric — "learning" is writing rows to Qdrant — and every background task runs off the request path, so the latency and reliability of normal requests are unaffected.

**Requirements:** a reachable Qdrant, an OpenAI-compatible **embeddings** endpoint (e.g. a small multilingual model served by [llama.cpp](https://github.com/ggml-org/llama.cpp) or Ollama), and a chat endpoint for the background learning LLM (defaults to the upstream; point `ANTHROPIC_PROXY_SKILLS_LLM_URL` at a no-auth internal backend to avoid spending a client key / user quota). Verification reuses the bundled open-websearch.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_PROXY_SKILLS_ENABLED` | `false` | Master switch for **injection** (the read path) |
| `ANTHROPIC_PROXY_SKILLS_LEARN` | `false` | Enable the **learning** background loops (distil + verify + curate) |
| `ANTHROPIC_PROXY_SKILLS_PROACTIVE` | `false` | Enable **proactive** research of asked questions (requires `LEARN`) |
| `ANTHROPIC_PROXY_SKILLS_QDRANT_URL` | `http://qdrant:6333` | Qdrant base URL |
| `ANTHROPIC_PROXY_SKILLS_COLLECTION` | `skills` | Qdrant collection name |
| `ANTHROPIC_PROXY_SKILLS_EMBED_URL` | upstream `/embeddings` | OpenAI-compatible embeddings endpoint |
| `ANTHROPIC_PROXY_SKILLS_EMBED_MODEL` | – | Embedding model name (empty disables retrieval) |
| `ANTHROPIC_PROXY_SKILLS_LLM_URL` | upstream chat URL | Chat endpoint for background learning calls; when set, called with **no auth** |
| `ANTHROPIC_PROXY_SKILLS_LLM_MODEL` | `auto` | Model for the background learning/judge calls |
| `ANTHROPIC_PROXY_SKILLS_API_KEY` | – | Key for background calls not tied to a client request (falls back to the last-seen client key) |
| `ANTHROPIC_PROXY_SKILLS_TOP_K` | `3` | Max skills injected per request |
| `ANTHROPIC_PROXY_SKILLS_MIN_SCORE` | `0.5` | Minimum cosine score to inject (filters weak matches; ~`0.45` suits bge-m3) |
| `ANTHROPIC_PROXY_SKILLS_INJECT_TIERS` | `verified,trusted` | Tiers eligible for injection (candidates excluded) |
| `ANTHROPIC_PROXY_SKILLS_VERIFY_INTERVAL_SECS` | `300` | Verification-loop interval |
| `ANTHROPIC_PROXY_SKILLS_SOAK_SECS` | `1209600` | Soak before `verified` → `trusted` (default 14 days) |
| `ANTHROPIC_PROXY_SKILLS_CURATE_INTERVAL_SECS` | `600` | Curation-loop interval |
| `ANTHROPIC_PROXY_SKILLS_PROACTIVE_INTERVAL_SECS` | `600` | Proactive-loop interval |
| `ANTHROPIC_PROXY_SKILLS_RETENTION_DAYS` | `30` | Drop unverified candidates older than this |

> **Safety.** Learning from the open web is a documented poisoning vector, so the trust gate is the load-bearing control: unverified knowledge is never injected, promotion requires independent multi-source corroboration rather than the model's confidence, and the web-reading LLM is quarantined (no tools / no write access). Treat your embeddings / LLM / Qdrant endpoints as trusted infrastructure.

Example (Docker Compose) — co-located Qdrant + a llama.cpp embedding server, with learning routed at a no-auth internal backend:

```yaml
  qdrant:
    image: qdrant/qdrant
  embeddings:                       # OpenAI-compatible /v1/embeddings (multilingual bge-m3)
    image: ghcr.io/ggml-org/llama.cpp:server
    command: ["-hf","gpustack/bge-m3-GGUF:Q4_K_M","--embeddings","--pooling","cls","--host","0.0.0.0","--port","8080"]
  anthropic-proxy:
    environment:
      ANTHROPIC_PROXY_SKILLS_ENABLED: "true"
      ANTHROPIC_PROXY_SKILLS_LEARN: "true"
      ANTHROPIC_PROXY_SKILLS_PROACTIVE: "true"
      ANTHROPIC_PROXY_SKILLS_QDRANT_URL: "http://qdrant:6333"
      ANTHROPIC_PROXY_SKILLS_EMBED_URL: "http://embeddings:8080/v1/embeddings"
      ANTHROPIC_PROXY_SKILLS_EMBED_MODEL: "bge-m3"
      ANTHROPIC_PROXY_SKILLS_LLM_URL: "http://your-backend:8000/v1/chat/completions"
      ANTHROPIC_PROXY_SKILLS_LLM_MODEL: "your-model"
      ANTHROPIC_PROXY_SKILLS_MIN_SCORE: "0.45"
```

## Supported Features

✅ Text messages
✅ System prompts (single and multiple)
✅ Image content (base64)
✅ Tool/function calling + tool results (multi-turn `tool_use` ↔ `tool_result` round-trip)
✅ `tool_choice` — `auto` / `any` / `tool` / `none`, incl. `disable_parallel_tool_use`
✅ Streaming responses (SSE; handles `\n\n` and `\r\n\r\n` framing)
✅ Extended thinking (automatic model routing; `reasoning_content` preserved in both streaming and non-streaming)
✅ Server-side `web_search` / `web_fetch` emulation (runs the loop against a bundled open-websearch; faithful `server_tool_use` + `*_tool_result` blocks; streaming + non-streaming; SSE heartbeats) — see [Web search & fetch](#web-search--fetch)
✅ `metadata.user_id` (forwarded as OpenAI `user`)
✅ `refusal` stop reason (mapped from upstream `content_filter`)
✅ Stop sequences, `max_tokens`, `temperature`, `top_p`
✅ `POST /v1/messages/count_tokens` — exact via the upstream `/tokenize` (when `ANTHROPIC_PROXY_UPSTREAM_TOKENIZE=true`), else a local BPE estimate; also accepts `?beta=true`
✅ Streaming usage — real `input_tokens` / `output_tokens` / `cache_read_input_tokens` surfaced in `message_delta` (captured from the upstream's final usage chunk), with an upfront estimate in `message_start`
✅ Prompt-cache token accounting — upstream `prompt_tokens_details.cached_tokens` is reported as Anthropic `cache_read_input_tokens` (and excluded from `input_tokens`), so Claude Code's cache/cost stats are accurate

> `top_k` is accepted but not forwarded — Chat Completions has no equivalent.
>
> **Note:** Make sure your upstream model supports tool use, especially for coding agents like Claude Code.

## Reliability

- **Upstream status codes are preserved.** A client error (`400` invalid request, `401`/`403`/`404`, `413`, `429` rate limit) is surfaced with its original status and an Anthropic-shaped error body — `{"type":"error","error":{"type":...,"message":...}}` — instead of being masked as a generic `502`. Only genuine transport failures (no HTTP response) map to `502`.
- **Transient failures are retried.** Connection / timeout / body-read errors and retriable statuses (`429`, `5xx`) are retried up to 3× per upstream URL with a fresh connection.
- **Context-overflow auto-recovery.** If the upstream rejects a request because `input + max_tokens` exceeds the model's context window, the proxy re-tokenizes the actual request to learn the true input size (taking the larger of that and the error's own lower bound, so it always converges), clamps `max_tokens` to fit, and retries — so a conversation that's *just* over the limit (and even Claude Code's `/compact`, which otherwise dead-locks because it also requests output) completes instead of hard-failing on a `400`.
- **Stale-connection hardening.** Pooled idle connections are kept short-lived with TCP keep-alive, eliminating the intermittent `502`s caused by reusing a socket the upstream silently closed.
- **600 s request timeout**, matching a typical fronting nginx `proxy_read_timeout`, so long generations and streams are not cut short.

## Behind a reverse proxy (nginx)

When running this proxy behind nginx/OpenResty, for the location that forwards to it:

```nginx
location / {
    proxy_pass http://anthropic-proxy:3000;

    # SSE streaming
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_request_buffering off;

    # Let the proxy's JSON error bodies pass through unchanged
    proxy_intercept_errors off;

    # Align with the proxy's 600 s request timeout
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
    send_timeout 600s;
}
```

- Keep **`proxy_intercept_errors off`** so the proxy's Anthropic-shaped JSON errors reach the client (otherwise `error_page` replaces them with plain text).
- Keep **buffering off** for streaming.
- Align timeouts to **600 s** (or raise the proxy's accordingly) so neither side truncates long responses.

## Known Limitations

The following Anthropic API features are **not supported** (Claude Code and similar tools work fine without them):

- `service_tier` parameter (no portable OpenAI-compatible equivalent)
- `context_management` parameter (Anthropic server-side feature; no upstream equivalent)
- `container` parameter (Anthropic code-execution sandbox; no upstream equivalent)
- Inline citations — emulated `web_search`/`web_fetch` return result blocks, but without inline `web_search_result_location` citation markers
- `pause_turn` stop reason — the proxy runs the emulated web-tool loop to completion (kept alive with heartbeats) and returns `end_turn`, so it never emits `pause_turn`
- Message Batches API
- Files API
- Admin API

Without `ANTHROPIC_PROXY_UPSTREAM_TOKENIZE`, `count_tokens` returns a **local BPE estimate** rather than an exact count. Enable it to get exact counts from a vLLM-style `/tokenize` endpoint.

## Troubleshooting

**`UPSTREAM_BASE_URL is required`** — set the upstream endpoint, e.g. `https://openrouter.ai/api`, `https://api.openai.com`, or `http://localhost:11434`.

**`405 Method Not Allowed` / wrong upstream path** — check how `UPSTREAM_BASE_URL` resolves (see the forms above). Partial paths like `.../chat` and URLs with query strings/fragments are rejected.

**Model not found** — set `REASONING_MODEL` / `COMPLETION_MODEL`, or use `ANTHROPIC_PROXY_MODEL_MAP` to remap client model names to upstream ones. Ensure the model IDs you advertise to clients match the map keys.

**Gateway/WAF blocks Claude Code prompts with `403`** — use `ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS` / `--system-prompt-ignore` to strip offending terms.

**Intermittent `502` under load** — ensure you are on a current build (stale-connection hardening + retries). If a 502 persists, the upstream itself returned a transport failure; check the proxy logs (`--debug`).

## Development

```bash
cargo test          # unit tests
cargo clippy        # lints
cargo fmt           # formatting
```

## License

MIT License — Copyright (c) 2025 m0n0x41d (Ivan Zakutnii). Fork maintained by ExpTech. See [LICENSE](LICENSE).

## Links

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Rust Documentation](https://doc.rust-lang.org/)
