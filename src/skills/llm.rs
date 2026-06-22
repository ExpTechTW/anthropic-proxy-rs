//! Background LLM calls for the learning pipeline (distillation, verification, proactive study).
//!
//! Routes through the same OpenAI-compatible upstream as the proxy, using the configured skills
//! model. These run off the request path (background tasks/loops), so failures are swallowed —
//! the worst case is "we didn't learn this time", never a broken proxied request.

use crate::config::Config;
use crate::models::openai;
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const LLM_TIMEOUT: Duration = Duration::from_secs(120);

// ── Tiered routing for HARD background tasks (difficulty grading) ──────────────────────────────
// Easy tasks (yes/no checks) stay on the self-hosted `auto` backend. Hard synthesis tasks tier up
// to a strong FREE model on OpenRouter (nemotron), rate-limited to spread the daily free quota;
// after repeated failures they fail over ONCE to a cheap paid backup (gemini-2.5-flash-lite) before
// resetting back to nemotron. The backup does NOT use paid native grounding — it judges the free
// SearXNG evidence the proxy already put in the prompt. Anything rate-limited/failed still completes
// on `auto`, so a hard task never goes unlearned.
const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
/// Shorter than `LLM_TIMEOUT`: a saturated free model (nemotron's free tier 504s after ~300s on
/// OpenRouter's side) must fail FAST so the background task falls back to `auto` instead of stalling.
/// A healthy 550B still answers well within this.
const OPENROUTER_TIMEOUT: Duration = Duration::from_secs(60);
const HARD_MODEL: &str = "nvidia/nemotron-3-ultra-550b-a55b:free";
const BACKUP_MODEL: &str = "google/gemini-2.5-flash-lite";
const NEM_GAP: Duration = Duration::from_secs(120); // nemotron: at most one call per 2 min
const GEM_GAP: Duration = Duration::from_secs(600); // gemini (paid backup): at most one per 10 min
const FAIL_THRESHOLD: u32 = 5; // nemotron failures before switching to the backup once

#[derive(Default)]
struct Router {
    nem_last: Option<Instant>,
    nem_fails: u32,
    gem_last: Option<Instant>,
}

enum Route {
    Auto,
    Nemotron,
    Gemini,
}

fn router() -> &'static Mutex<Router> {
    static R: OnceLock<Mutex<Router>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Router::default()))
}

/// Pick the backend for a HARD task, honouring per-model rate limits + the failover counter.
fn pick_hard_route() -> Route {
    let now = Instant::now();
    let mut r = router().lock().unwrap();
    if r.nem_fails >= FAIL_THRESHOLD {
        // 5 nemotron failures → switch to the paid backup ONCE, then reset and re-accumulate.
        if r.gem_last.map_or(true, |t| now.duration_since(t) >= GEM_GAP) {
            r.gem_last = Some(now);
            r.nem_fails = 0;
            return Route::Gemini;
        }
        return Route::Auto; // backup still rate-limited → self-host this one
    }
    if r.nem_last.map_or(true, |t| now.duration_since(t) >= NEM_GAP) {
        r.nem_last = Some(now);
        return Route::Nemotron;
    }
    Route::Auto // nemotron rate-limited (≤1 / NEM_GAP) → self-host the overflow
}

fn report_nemotron(success: bool) {
    let mut r = router().lock().unwrap();
    if success {
        r.nem_fails = 0;
    } else {
        r.nem_fails = r.nem_fails.saturating_add(1);
    }
}

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
    // Prefer the configured background endpoint so learning runs off the client's key/quota. When
    // that endpoint needs auth (e.g. routing through the token-accounting upstream on :9000), the
    // configured skills key is sent for ALL learning calls (distill/verify/proactive) uniformly; a
    // no-auth internal backend simply leaves it unset → no header. Otherwise fall back to the authed
    // upstream + the caller's key.
    let (url, auth) = match &config.skills.llm_url {
        Some(u) => (u.clone(), config.skills.api_key.as_deref()),
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

/// One OpenRouter chat turn → assistant text. Uses the configured OpenRouter key. `web_search`
/// turns on the web plugin with `engine: "native"` → the model provider's OWN search (for the
/// gemini backup that means Gemini's native Google Search grounding, not OpenRouter's Exa plugin).
/// `None` on failure.
async fn openrouter_chat(
    config: &Config,
    client: &Client,
    model: &str,
    system: &str,
    user: &str,
    max_tokens: u32,
    web_search: bool,
) -> Option<String> {
    let key = config.skills.openrouter_key.as_deref()?;
    let mut body = json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": false,
    });
    if web_search {
        // engine:"native" → the provider's built-in search (Gemini → Google Search grounding).
        body["plugins"] = json!([{"id": "web", "engine": "native"}]);
    }
    let resp = client
        .post(OPENROUTER_URL)
        .timeout(OPENROUTER_TIMEOUT)
        .header("Authorization", format!("Bearer {key}"))
        .header("X-Title", "anthropic-proxy-skills")
        .json(&body)
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        tracing::warn!(model = %model, "skills/llm: openrouter returned {}", resp.status());
        return None;
    }
    let parsed: openai::OpenAIResponse = resp.json().await.ok()?;
    parsed.choices.into_iter().next()?.message.content
}

/// HARD task path: tier up to OpenRouter (nemotron → gemini) per the rate-limit/failover router,
/// always falling back to the self-hosted `auto` backend so the task still completes. With no
/// OpenRouter key configured, behaves exactly like `chat` on `auto`.
async fn chat_hard(
    config: &Config,
    client: &Client,
    system: &str,
    user: &str,
    max_tokens: u32,
) -> Option<String> {
    let auto_key = config.skills.api_key.as_deref();
    if config.skills.openrouter_key.is_none() {
        return chat(config, client, system, user, auto_key, max_tokens).await;
    }
    match pick_hard_route() {
        Route::Nemotron => {
            let out =
                openrouter_chat(config, client, HARD_MODEL, system, user, max_tokens, false).await;
            report_nemotron(out.is_some());
            match out {
                Some(t) => Some(t),
                None => chat(config, client, system, user, auto_key, max_tokens).await,
            }
        }
        Route::Gemini => {
            // web_search=false: no paid native grounding — the prompt already carries the proxy's
            // free SearXNG evidence, so gemini-2.5-flash-lite just judges it (≈token-cost only).
            match openrouter_chat(config, client, BACKUP_MODEL, system, user, max_tokens, false).await
            {
                Some(t) => Some(t),
                None => chat(config, client, system, user, auto_key, max_tokens).await,
            }
        }
        Route::Auto => chat(config, client, system, user, auto_key, max_tokens).await,
    }
}

/// `chat_hard` + JSON extraction — the HARD-task counterpart of [`chat_json`].
pub async fn chat_json_hard(
    config: &Config,
    client: &Client,
    system: &str,
    user: &str,
    max_tokens: u32,
) -> Option<Value> {
    let text = chat_hard(config, client, system, user, max_tokens).await?;
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
