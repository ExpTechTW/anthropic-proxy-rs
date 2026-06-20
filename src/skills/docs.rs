//! Docs push-injection (streaming-preserving alternative to the pull tool loop).
//!
//! When the user's query mentions a library that docs-mcp has indexed, fetch a short doc snippet
//! and inject it as a system block before forwarding — so the request still streams normally.
//! Best-effort: any failure injects nothing and the request proceeds untouched. Library names are
//! matched as whole words against the cached `list_libraries`, so unindexed mentions cost nothing.

use crate::config::Config;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const LIB_TTL: Duration = Duration::from_secs(300);
const MAX_LIBS: usize = 2;
const SNIPPET_CHARS: usize = 1500;
const MCP_TIMEOUT: Duration = Duration::from_secs(20);

fn lib_cache() -> &'static Mutex<Option<(Vec<String>, Instant)>> {
    static C: OnceLock<Mutex<Option<(Vec<String>, Instant)>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(None))
}

/// A docs snippet for indexed libraries named in `query`, or `None` if none apply / on failure.
pub async fn relevant_docs(config: &Config, client: &Client, query: &str) -> Option<String> {
    let url = config.skills.docs_mcp_url.as_ref()?;
    if query.trim().is_empty() {
        return None;
    }
    let libs = libraries(client, url).await;
    if libs.is_empty() {
        return None;
    }
    let words: HashSet<String> = query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
        .map(str::to_string)
        .collect();
    let matched: Vec<&String> = libs
        .iter()
        .filter(|l| l.len() >= 3 && words.contains(&l.to_lowercase()))
        .take(MAX_LIBS)
        .collect();
    if matched.is_empty() {
        return None;
    }
    let mut out = String::new();
    for lib in matched {
        if let Some(text) = mcp_text(
            client,
            url,
            "search_docs",
            json!({"library": lib, "query": query, "limit": 2}),
        )
        .await
        {
            let snippet: String = text.chars().take(SNIPPET_CHARS).collect();
            out.push_str(&format!("## {lib}\n{snippet}\n\n"));
        }
    }
    if out.trim().is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Indexed library names (cached for `LIB_TTL`), parsed from docs-mcp `list_libraries` (lines
/// of the form `- name`).
async fn libraries(client: &Client, url: &str) -> Vec<String> {
    {
        let c = lib_cache().lock().unwrap();
        if let Some((libs, at)) = c.as_ref() {
            if at.elapsed() < LIB_TTL {
                return libs.clone();
            }
        }
    }
    let libs = mcp_text(client, url, "list_libraries", json!({}))
        .await
        .map(|t| {
            t.lines()
                .filter_map(|l| l.trim().strip_prefix("- ").map(|s| s.trim().to_string()))
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    *lib_cache().lock().unwrap() = Some((libs.clone(), Instant::now()));
    libs
}

/// One stateless MCP `tools/call`, returning `result.content[0].text`. Best-effort → `None`.
async fn mcp_text(client: &Client, url: &str, name: &str, arguments: Value) -> Option<String> {
    let body = json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments}
    });
    let resp = client
        .post(url)
        .timeout(MCP_TIMEOUT)
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream")
        .json(&body)
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let text = resp.text().await.ok()?;
    let data = text
        .lines()
        .find_map(|l| l.strip_prefix("data: "))
        .unwrap_or_else(|| text.trim());
    let v: Value = serde_json::from_str(data).ok()?;
    v.pointer("/result/content/0/text")
        .and_then(Value::as_str)
        .map(str::to_string)
}
