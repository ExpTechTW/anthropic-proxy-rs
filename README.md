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
- **Resilient** — retries transient failures, preserves upstream status codes, 600 s request timeout
- **Drop-in** — works with the official Anthropic SDKs and Claude Code

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/messages` | Main completion endpoint (streaming + non-streaming). Also accepts a trailing slash. |
| `POST` | `/v1/messages/count_tokens` | Local heuristic token estimate (upstreams expose no count endpoint) |
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

## Use with Claude Code

```bash
# Start the proxy as a daemon and launch Claude Code against it
anthropic-proxy --daemon && ANTHROPIC_BASE_URL=http://localhost:3000 claude

# Or in separate terminals:
anthropic-proxy                                   # terminal 1
ANTHROPIC_BASE_URL=http://localhost:3000 claude   # terminal 2
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

## Supported Features

✅ Text messages
✅ System prompts (single and multiple)
✅ Image content (base64)
✅ Tool/function calling + tool results (multi-turn `tool_use` ↔ `tool_result` round-trip)
✅ `tool_choice` — `auto` / `any` / `tool` / `none`, incl. `disable_parallel_tool_use`
✅ Streaming responses (SSE; handles `\n\n` and `\r\n\r\n` framing)
✅ Extended thinking (automatic model routing; `reasoning_content` preserved in both streaming and non-streaming)
✅ `metadata.user_id` (forwarded as OpenAI `user`)
✅ `refusal` stop reason (mapped from upstream `content_filter`)
✅ Stop sequences, `max_tokens`, `temperature`, `top_p`
✅ `POST /v1/messages/count_tokens` (local heuristic estimate; also accepts `?beta=true`)
✅ Prompt-cache token accounting — upstream `prompt_tokens_details.cached_tokens` is reported as Anthropic `cache_read_input_tokens` (and excluded from `input_tokens`), so Claude Code's cache/cost stats are accurate

> `top_k` is accepted but not forwarded — Chat Completions has no equivalent.
>
> **Note:** Make sure your upstream model supports tool use, especially for coding agents like Claude Code.

## Reliability

- **Upstream status codes are preserved.** A client error (`400` invalid request, `401`/`403`/`404`, `413`, `429` rate limit) is surfaced with its original status and an Anthropic-shaped error body — `{"type":"error","error":{"type":...,"message":...}}` — instead of being masked as a generic `502`. Only genuine transport failures (no HTTP response) map to `502`.
- **Transient failures are retried.** Connection / timeout / body-read errors and retriable statuses (`429`, `5xx`) are retried up to 3× per upstream URL with a fresh connection.
- **Context-overflow auto-recovery.** If the upstream rejects a request because `input + max_tokens` exceeds the model's context window, the proxy clamps `max_tokens` to fit and retries once — so a conversation that's *just* over the limit (and even Claude Code's `/compact`, which otherwise dead-locks because it also requests output) can still complete instead of hard-failing on a `400`.
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
- Citations in responses
- `pause_turn` stop reason (only emitted by Anthropic server-side tool loops, which this proxy does not relay)
- Message Batches API
- Files API
- Admin API

`count_tokens` returns a **heuristic estimate** (~4 bytes/token), not an exact tokenizer count — upstreams expose no count endpoint.

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
