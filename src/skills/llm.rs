//! Background LLM calls for the learning pipeline (distillation, verification, proactive study).
//!
//! Routes through the same OpenAI-compatible upstream as the proxy, using the configured skills
//! model. These run off the request path (background tasks/loops), so failures are swallowed —
//! the worst case is "we didn't learn this time", never a broken proxied request.

use crate::config::Config;
use crate::models::openai;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;

const LLM_TIMEOUT: Duration = Duration::from_secs(120);

/// One non-streaming chat turn → the assistant's text. `temperature` is pinned to 0 for
/// deterministic judging; reasoning is forced low (and qwen's `/no_think` appended) because the
/// learning tasks are mechanical and don't need deep chain-of-thought. `None` on any failure.
pub async fn chat(
    config: &Config,
    client: &Client,
    system: &str,
    user: &str,
    api_key: Option<&str>,
    max_tokens: u32,
) -> Option<String> {
    // Prefer the configured background endpoint (e.g. a no-auth internal backend) so background
    // tasks need no client key; otherwise the authed upstream + the provided key.
    let (url, auth) = match &config.skills.llm_url {
        Some(u) => (u.clone(), None),
        None => (config.chat_completions_urls().into_iter().next()?, api_key),
    };
    let body = json!({
        "model": config.skills.llm_model,
        "messages": [
            {"role": "system", "content": format!("{system}\n/no_think")},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "low",
        "stream": false,
    });
    let mut rb = client.post(&url).timeout(LLM_TIMEOUT).json(&body);
    if let Some(key) = auth {
        rb = rb.header("Authorization", format!("Bearer {key}"));
    }
    let resp = rb.send().await.ok()?;
    if !resp.status().is_success() {
        tracing::warn!("skills/llm: chat returned {}", resp.status());
        return None;
    }
    let parsed: openai::OpenAIResponse = resp.json().await.ok()?;
    parsed.choices.into_iter().next()?.message.content
}

/// `chat` + best-effort JSON extraction (tolerates ```json fences, `<think>` blocks, leading
/// prose). `None` if no balanced JSON object can be recovered.
pub async fn chat_json(
    config: &Config,
    client: &Client,
    system: &str,
    user: &str,
    api_key: Option<&str>,
    max_tokens: u32,
) -> Option<Value> {
    let text = chat(config, client, system, user, api_key, max_tokens).await?;
    extract_json(&text)
}

/// Recover the first balanced JSON object from a model response.
fn extract_json(text: &str) -> Option<Value> {
    // Drop any leaked chain-of-thought.
    let text = match text.rfind("</think>") {
        Some(i) => &text[i + "</think>".len()..],
        None => text,
    };
    if let Ok(v) = serde_json::from_str::<Value>(text.trim()) {
        return Some(v);
    }
    let bytes = text.as_bytes();
    let start = text.find('{')?;
    let (mut depth, mut in_str, mut esc) = (0usize, false, false);
    for i in start..bytes.len() {
        let c = bytes[i] as char;
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
        } else {
            match c {
                '"' => in_str = true,
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return serde_json::from_str(&text[start..=i]).ok();
                    }
                }
                _ => {}
            }
        }
    }
    None
}
