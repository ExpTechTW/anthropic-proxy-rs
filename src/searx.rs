//! Minimal SearXNG JSON-API client for the `web_search` emulation.
//!
//! SearXNG is a self-hosted metasearch engine that aggregates 70+ upstream engines and dedupes /
//! ranks across them. Unlike `open-websearch` it speaks a plain HTTP JSON API rather than MCP:
//!   GET /search?q=…&format=json  →  {"results": [{"title","url","content",…}], …}
//!
//! We map each result's `content` (the snippet) onto [`SearchResult::description`] so the agent
//! loop renders and feeds results identically regardless of which backend produced them. SearXNG
//! has no page-fetch tool, so `web_fetch` stays on open-websearch (see [`crate::websearch`]).
//!
//! Egress: when the host can't reach the engines directly, configure SearXNG's
//! `outgoing.proxies` (httpx, supports SOCKS5) — the proxy here only talks to SearXNG.

use crate::websearch::SearchResult;
use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::time::Duration;

/// Per-search timeout. SearXNG fans out to many engines but caps itself; keep this aligned with
/// the open-websearch client so a slow search fails fast and the agent loop continues without it.
const SEARCH_TIMEOUT: Duration = Duration::from_secs(25);

/// One SearXNG JSON result (the subset we need; other fields like `engine`/`score` are ignored).
#[derive(Debug, Deserialize)]
struct SearxResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    content: String,
}

/// Client for one SearXNG instance (e.g. `http://searxng:8080`).
pub struct SearxClient {
    base: String,
    http: Client,
}

impl SearxClient {
    pub fn new(base: impl Into<String>, http: Client) -> Self {
        Self {
            base: base.into(),
            http,
        }
    }

    /// Run a web search via SearXNG's JSON API and return up to `limit` results. The instance
    /// must allow the `json` output format (`search.formats: [html, json]` in `settings.yml`).
    pub async fn search(&self, query: &str, limit: u32) -> Result<Vec<SearchResult>> {
        let url = format!("{}/search", self.base.trim_end_matches('/'));
        let resp = self
            .http
            .get(&url)
            .timeout(SEARCH_TIMEOUT)
            .header("accept", "application/json")
            .query(&[
                ("q", query),
                ("format", "json"),
                ("safesearch", "0"),
            ])
            .send()
            .await
            .context("searxng request failed")?;
        if !resp.status().is_success() {
            return Err(anyhow!("searxng returned {}", resp.status()));
        }
        let body: Value = resp.json().await.context("parsing searxng json")?;
        let results = body
            .get("results")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        Ok(results
            .into_iter()
            .filter_map(|r| serde_json::from_value::<SearxResult>(r).ok())
            .filter(|r| !r.url.is_empty())
            .take(limit as usize)
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                description: r.content,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live test — requires a SearXNG with the JSON format enabled at $SEARXNG_URL.
    /// Run with: `SEARXNG_URL=http://localhost:8080 cargo test searx -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore]
    async fn live_search_via_searxng() {
        let base = std::env::var("SEARXNG_URL").expect("set SEARXNG_URL");
        let client = SearxClient::new(base, Client::new());
        let results = client
            .search("anthropic claude api", 5)
            .await
            .expect("search should succeed");
        assert!(!results.is_empty(), "expected at least one result");
        for r in &results {
            println!("- {} | {}", r.title, r.url);
            assert!(!r.url.is_empty());
        }
    }
}
