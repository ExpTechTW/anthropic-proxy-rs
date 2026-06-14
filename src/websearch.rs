//! Minimal MCP (Streamable HTTP) client for the self-hosted `open-websearch` server.
//!
//! `open-websearch` exposes its `search` tool only over MCP at `/mcp` (no plain REST in the
//! current build), so we speak just enough of the protocol to call one tool:
//!   1. POST `initialize`               → read the `Mcp-Session-Id` response header
//!   2. POST `notifications/initialized`
//!   3. POST `tools/call` (`search`)    → parse the SSE `data:` line → JSON-RPC result
//!
//! Responses come back as Server-Sent Events (`event: message\ndata: {json}`), so each call
//! reads the body and extracts the first `data:` line. The `search` result's
//! `content[0].text` is itself a JSON string of `{results: [{title,url,description,…}], …}`.
//!
//! This backs the emulation of Anthropic's server-side browsing tools against local
//! OpenAI-compatible models, which can't browse: `web_search` → the `search` tool,
//! `web_fetch` → the `fetchWebContent` tool.

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;

/// Per-MCP-call timeout. The shared HTTP client allows 600s (sized for long generations), but a
/// search/fetch that hangs (e.g. DuckDuckGo anti-bot stalls in request mode) must fail fast so
/// the agent loop doesn't freeze the whole request — the model then continues without it.
const RPC_TIMEOUT: Duration = Duration::from_secs(25);

/// A single web result (the subset the model needs). Extra fields (`source`, `engine`) are
/// ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchResult {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub description: String,
}

/// The fetched content of one page (`fetchWebContent`). `final_url` follows redirects; the
/// `content` is plain-text/Markdown already truncated to `maxChars` by the server.
#[derive(Debug, Clone, Deserialize)]
pub struct FetchResult {
    /// The post-redirect URL; we surface this as the canonical fetched URL.
    #[serde(rename = "finalUrl", default)]
    pub final_url: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub content: String,
}

/// Client for one `open-websearch` MCP endpoint (e.g. `http://localhost:3100/mcp`).
pub struct WebSearchClient {
    endpoint: String,
    http: Client,
}

impl WebSearchClient {
    pub fn new(endpoint: impl Into<String>, http: Client) -> Self {
        Self {
            endpoint: endpoint.into(),
            http,
        }
    }

    /// POST one JSON-RPC message; return `(parsed_envelope, session_id_from_header)`.
    /// The body is SSE — we take the first `data:` line (falling back to plain JSON).
    async fn rpc(&self, body: Value, session: Option<&str>) -> Result<(Value, Option<String>)> {
        let mut req = self
            .http
            .post(&self.endpoint)
            .timeout(RPC_TIMEOUT)
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream");
        if let Some(s) = session {
            req = req.header("mcp-session-id", s);
        }
        let resp = req
            .json(&body)
            .send()
            .await
            .context("open-websearch request failed")?;
        let session_id = resp
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let text = resp.text().await.context("reading open-websearch body")?;
        let data = text
            .lines()
            .find_map(|l| l.strip_prefix("data: "))
            .unwrap_or_else(|| text.trim());
        if data.is_empty() {
            return Ok((Value::Null, session_id));
        }
        let value = serde_json::from_str(data)
            .with_context(|| format!("parsing open-websearch response: {data}"))?;
        Ok((value, session_id))
    }

    /// Run one `tools/call` end to end (initialize → initialized → call) and return the
    /// tool's inner payload — `result.content[0].text` parsed as JSON, which is how
    /// open-websearch wraps both `search` and `fetchWebContent` results.
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value> {
        // 1. initialize → session id
        let (_init, session_id) = self
            .rpc(
                json!({
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "anthropic-proxy", "version": "1"}
                    }
                }),
                None,
            )
            .await?;
        let session_id =
            session_id.ok_or_else(|| anyhow!("open-websearch did not return an Mcp-Session-Id"))?;

        // 2. initialized notification (no response body expected)
        self.rpc(
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            Some(&session_id),
        )
        .await?;

        // 3. tools/call
        let (resp, _) = self
            .rpc(
                json!({
                    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": name, "arguments": arguments}
                }),
                Some(&session_id),
            )
            .await?;

        // A tool-level failure comes back as result.isError with the message in content[0].text.
        if resp.pointer("/result/isError") == Some(&Value::Bool(true)) {
            let msg = resp
                .pointer("/result/content/0/text")
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            return Err(anyhow!("open-websearch {name} error: {msg}"));
        }

        let text = resp
            .pointer("/result/content/0/text")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("unexpected {name} response: {resp}"))?;
        serde_json::from_str(text).with_context(|| format!("parsing inner {name} JSON"))
    }

    /// Run a web search via the `search` MCP tool. `engines` empty → server default;
    /// `mode` is `"request"` (fast HTTP), `"auto"`, or `"playwright"`.
    pub async fn search(
        &self,
        query: &str,
        limit: u32,
        engines: &[String],
        mode: &str,
    ) -> Result<Vec<SearchResult>> {
        let mut arguments = json!({"query": query, "limit": limit, "searchMode": mode});
        if !engines.is_empty() {
            arguments["engines"] = json!(engines);
        }
        let inner = self.call_tool("search", arguments).await?;
        let results = inner
            .get("results")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        Ok(results
            .into_iter()
            .filter_map(|r| serde_json::from_value(r).ok())
            .collect())
    }

    /// Fetch a single URL's content via the `fetchWebContent` MCP tool. `max_chars` caps the
    /// returned text (the server truncates); ~30K chars ≈ Anthropic's default fetch budget.
    pub async fn fetch(&self, url: &str, max_chars: u32) -> Result<FetchResult> {
        let inner = self
            .call_tool(
                "fetchWebContent",
                json!({"url": url, "maxChars": max_chars}),
            )
            .await?;
        serde_json::from_value(inner).context("parsing fetchWebContent result")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live test — requires `MODE=http PORT=3100 npx open-websearch@latest` running.
    /// Run with: `cargo test websearch -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore]
    async fn live_search_via_open_websearch() {
        let client = WebSearchClient::new("http://localhost:3100/mcp", Client::new());
        let results = client
            .search(
                "anthropic claude api",
                3,
                &["duckduckgo".to_string()],
                "request",
            )
            .await
            .expect("search should succeed");
        assert!(!results.is_empty(), "expected at least one result");
        for r in &results {
            println!("- {} | {}", r.title, r.url);
            assert!(!r.url.is_empty());
        }
    }
}
