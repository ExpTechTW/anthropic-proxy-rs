//! Embeddings for skill retrieval, via an OpenAI-compatible `/v1/embeddings` endpoint.
//!
//! Reuses the proxy's upstream (or an explicit override) so no separate embedding service is
//! required. Best-effort: any failure (no model configured, endpoint down, bad body) returns
//! `None`, and the caller simply skips injection for that request.

use crate::config::Config;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;

/// A retrieval embedding must be quick; the query is short and this sits on the request path.
const EMBED_TIMEOUT: Duration = Duration::from_secs(20);
/// Cap the text we embed — the routing signal is the user's ask, not a whole pasted file, and a
/// huge input both slows the call and dilutes the vector.
const MAX_QUERY_CHARS: usize = 2000;

#[derive(Deserialize)]
struct EmbeddingsResponse {
    #[serde(default)]
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    #[serde(default)]
    embedding: Vec<f32>,
}

/// Embed `text` for retrieval. Returns `None` when no embedding model is configured or the call
/// fails — never an error (skills are an enhancement, never a request-breaker).
pub async fn embed(
    config: &Config,
    client: &Client,
    text: &str,
    api_key: Option<&str>,
) -> Option<Vec<f32>> {
    if config.skills.embed_model.is_empty() {
        tracing::debug!("skills: no embed model configured; skipping retrieval");
        return None;
    }
    let url = config.skills_embed_url()?;
    let input: String = text.chars().take(MAX_QUERY_CHARS).collect();

    let mut rb = client
        .post(&url)
        .timeout(EMBED_TIMEOUT)
        .json(&json!({ "model": config.skills.embed_model, "input": input }));
    if let Some(key) = api_key {
        rb = rb.header("Authorization", format!("Bearer {key}"));
    }

    let resp = match rb.send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("skills: embed request failed ({url}): {e}");
            return None;
        }
    };
    if !resp.status().is_success() {
        tracing::warn!("skills: embed endpoint returned {}", resp.status());
        return None;
    }
    let parsed: EmbeddingsResponse = match resp.json().await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("skills: embed response unparseable: {e}");
            return None;
        }
    };
    let vector = parsed.data.into_iter().next().map(|d| d.embedding);
    match &vector {
        Some(v) if !v.is_empty() => vector,
        _ => {
            tracing::warn!("skills: embed response contained no vector");
            None
        }
    }
}
