use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::metrics;
use crate::models::{anthropic, openai};
use crate::skills;
use crate::translate::{core, pipeline, stream, websearch_agent};
use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Extension, Json,
};
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use reqwest::Client;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub async fn proxy_handler(
    Extension(config): Extension<Arc<Config>>,
    Extension(client): Extension<Client>,
    headers: HeaderMap,
    Json(mut req): Json<anthropic::AnthropicRequest>,
) -> ProxyResult<Response> {
    let is_streaming = req.stream.unwrap_or(false);
    let start = Instant::now();
    let client_model = req.model.clone();

    let api_key = resolve_api_key(&config, &headers);
    skills::remember_api_key(api_key.as_deref());

    let user_text = skills::last_user_text(&req).unwrap_or_default();
    // Stage 5: record the asked question for proactive research (no-op unless proactive enabled).
    if config.skills.proactive {
        skills::record_question(&user_text);
    }

    // Auto-inject relevant learned skills (Stage 1). Best-effort: embedding/Qdrant failures
    // return no skills and the request proceeds untouched, so the feature can never break a call.
    let injected_skills = if config.skills.enabled {
        let found = skills::retrieve(&config, &client, &user_text, api_key.as_deref()).await;
        let ids = skills::inject(&mut req, &found);
        if !ids.is_empty() {
            tracing::info!(
                count = ids.len(),
                titles = %found.iter().map(|s| s.title.as_str()).collect::<Vec<_>>().join(" | "),
                "skills: injected into request"
            );
        }
        ids
    } else {
        Vec::new()
    };

    // Stage 2: learn from this conversation's history in the background (off the request path;
    // throttled per-conversation; no-op unless ANTHROPIC_PROXY_SKILLS_LEARN is set).
    skills::maybe_spawn_distill(config.clone(), client.clone(), &req, api_key.clone());

    tracing::debug!(model = %client_model, streaming = is_streaming, "received request");
    metrics::request_started(is_streaming);

    if config.log_requests {
        tracing::info!("request fields: {}", request_fields_summary(&req));
    }

    if config.verbose {
        tracing::trace!(
            "Incoming Anthropic request: {}",
            serde_json::to_string_pretty(&req).unwrap_or_default()
        );
    }

    // Emulate Anthropic's server-side web_search/web_fetch for models that can't browse:
    // detect those tools and, when a search-agent model is configured, rewrite them into
    // callable function tools and route to the agent loop. When unset, emulation is disabled —
    // strip the tools so the client gets an empty result instead of a broken/hanging search.
    let has_web_tools = websearch_agent::detect(&req);
    let use_websearch = has_web_tools && config.websearch_model.is_some();
    if use_websearch {
        websearch_agent::rewrite_tools(&mut req);
    } else if has_web_tools {
        websearch_agent::strip_web_tools(&mut req);
    }

    let policy = translation_policy(&config);
    let openai_req = pipeline::translate_request(req, &policy)?;
    let upstream_model = openai_req.model.clone();

    if config.verbose {
        tracing::trace!(
            "Transformed OpenAI request: {}",
            serde_json::to_string_pretty(&openai_req).unwrap_or_default()
        );
    }

    let result = if use_websearch {
        websearch_agent::handle(config, client, openai_req, api_key, is_streaming).await
    } else if is_streaming {
        handle_streaming(config, client, openai_req, api_key).await
    } else {
        handle_non_streaming(config, client, openai_req, api_key).await
    };

    let status = match &result {
        Ok(resp) => resp.status().as_u16(),
        Err(err) => err.status_code().as_u16(),
    };
    metrics::request_finished(start, status, is_streaming);

    // One line per request; failures log at WARN (visible at the default level)
    // with the upstream's error message so production issues are diagnosable.
    let elapsed_ms = start.elapsed().as_millis() as u64;
    match &result {
        Ok(_) => tracing::debug!(
            model = %client_model, upstream = %upstream_model,
            status, elapsed_ms, streaming = is_streaming, "request completed"
        ),
        Err(err) => tracing::warn!(
            model = %client_model, upstream = %upstream_model,
            status, elapsed_ms, streaming = is_streaming, "request failed: {err}"
        ),
    }

    // Transparency (#4): surface which learned skills were injected, without affecting behaviour.
    let mut result = result;
    if !injected_skills.is_empty() {
        if let Ok(resp) = result.as_mut() {
            if let Ok(value) = HeaderValue::from_str(&injected_skills.join(",")) {
                resp.headers_mut().insert("x-injected-skills", value);
            }
        }
    }

    result
}

/// Serialize a request to a compact JSON object with the bulky content blocks (`messages`,
/// `system`, `tools`) replaced by `*_count` breadcrumbs, keeping every other field (including
/// unknown ones flattened into `extra`). Used by the `ANTHROPIC_PROXY_LOG_REQUESTS` debug log
/// so newly-introduced client fields (e.g. `output_config.effort`) are visible per-request
/// without dumping conversation content — keeping each log line small.
fn request_fields_summary(req: &anthropic::AnthropicRequest) -> String {
    let mut value = serde_json::to_value(req).unwrap_or(serde_json::Value::Null);
    if let Some(obj) = value.as_object_mut() {
        // Replace each bulky field with a count so the line stays small but we can still see
        // how much was sent. `system` may be a string or an array of blocks.
        let messages = obj
            .remove("messages")
            .map_or(0, |m| m.as_array().map_or(0, Vec::len));
        // Keep each tool's `type` (server tools like `web_search_20250305`) or `name` (custom
        // tools) so we can see what the client actually requests — e.g. whether Claude Code
        // sends a server-side `web_search` tool we'd need to emulate — without dumping schemas.
        let tools_value = obj.remove("tools").unwrap_or(serde_json::Value::Null);
        let tool_ids: Vec<String> = tools_value
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|t| {
                        t.get("type")
                            .or_else(|| t.get("name"))
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("?")
                            .to_string()
                    })
                    .collect()
            })
            .unwrap_or_default();
        let tools = tools_value.as_array().map_or(0, Vec::len);
        let system = match obj.remove("system") {
            Some(serde_json::Value::Array(a)) => a.len(),
            Some(serde_json::Value::String(_)) => 1,
            _ => 0,
        };
        obj.insert("messages_count".to_string(), messages.into());
        obj.insert("tools_count".to_string(), tools.into());
        obj.insert("tools".to_string(), tool_ids.into());
        obj.insert("system_blocks".to_string(), system.into());
    }
    serde_json::to_string(&value).unwrap_or_default()
}

/// `POST /v1/messages/count_tokens` — Claude Code calls this for context budgeting.
/// Prefers the upstream `/tokenize` endpoint for an exact count (when enabled), and
/// falls back to a local BPE estimate when it's disabled, the model can't be tokenized,
/// or the upstream call fails.
pub async fn count_tokens_handler(
    Extension(config): Extension<Arc<Config>>,
    Extension(client): Extension<Client>,
    headers: HeaderMap,
    Json(req): Json<anthropic::CountTokensRequest>,
) -> ProxyResult<Response> {
    if config.verbose {
        tracing::trace!(
            "count_tokens request: {}",
            serde_json::to_string_pretty(&req).unwrap_or_default()
        );
    }

    let (input_tokens, source) = match upstream_count_tokens(&config, &client, &headers, &req).await
    {
        Some(count) => (count, "upstream"),
        None => (pipeline::estimate_input_tokens(&req), "estimate"),
    };

    tracing::debug!(
        model = %req.model,
        messages = req.messages.len(),
        tools = req.tools.as_ref().map_or(0, Vec::len),
        input_tokens,
        source,
        "count_tokens"
    );

    Ok(Json(serde_json::json!({ "input_tokens": input_tokens })).into_response())
}

/// Get an exact token count from the upstream `/tokenize`. Prefers the chat-aware form
/// (the gateway applies the model's chat template, so `count` already includes per-message
/// overhead — no estimate); falls back to plain-prompt tokenization plus our own overhead
/// for gateways without `messages` support. Returns `None` (→ local estimate) when
/// disabled, no model is given, or every attempt fails (failures are logged at WARN).
async fn upstream_count_tokens(
    config: &Config,
    client: &Client,
    headers: &HeaderMap,
    req: &anthropic::CountTokensRequest,
) -> Option<u32> {
    if !config.upstream_tokenize {
        return None;
    }

    // Translate the count request exactly like a real chat request, so the tokenized
    // messages match what the model is actually fed (system handling, tool_use/result,
    // images, tool schemas). select_model maps "auto"/model_map the same way chat does.
    let anth = anthropic::AnthropicRequest {
        model: req.model.clone(),
        messages: req.messages.clone(),
        max_tokens: 1,
        system: req.system.clone(),
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: None,
        stream: None,
        tools: req.tools.clone(),
        metadata: None,
        tool_choice: None,
        extra: serde_json::Value::Null,
    };
    let openai = pipeline::translate_request(anth, &translation_policy(config)).ok()?;
    if openai.model.is_empty() {
        return None;
    }

    let url = config.tokenize_urls().into_iter().next()?;
    let api_key = resolve_api_key(config, headers);

    // 1. Chat-aware tokenize: exact, template-applied count with no overhead guess.
    if let Some(count) = tokenize_chat(client, &url, &openai, api_key.as_deref()).await {
        return Some(count);
    }

    // 2. Fallback (gateway without messages support): plain prompt + our per-message overhead.
    tokenize_prompt(client, &url, &openai.model, req, api_key.as_deref()).await
}

/// Chat-aware `/tokenize`. Returns `None` so the caller falls back to the prompt form
/// when the gateway rejects `messages` (older build) or the response is unusable. Stays
/// quiet on failure — the prompt fallback logs if it fails too.
async fn tokenize_chat(
    client: &Client,
    url: &str,
    openai: &openai::OpenAIRequest,
    api_key: Option<&str>,
) -> Option<u32> {
    let mut req_builder = client
        .post(url)
        .json(&openai::TokenizeMessagesRequest {
            model: openai.model.clone(),
            messages: openai.messages.clone(),
            tools: openai.tools.clone(),
        })
        .timeout(Duration::from_secs(30));
    if let Some(key) = api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
    }
    let response = req_builder.send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let tokenized: openai::TokenizeResponse = response.json().await.ok()?;
    Some(tokenized.count)
}

/// Plain-prompt `/tokenize` plus our per-message chat-template overhead. A concrete model
/// reached this point, so unexpected failures are logged at WARN before falling back.
async fn tokenize_prompt(
    client: &Client,
    url: &str,
    model: &str,
    req: &anthropic::CountTokensRequest,
    api_key: Option<&str>,
) -> Option<u32> {
    let (prompt, message_count) = pipeline::collect_tokenize_text(req);

    let mut req_builder = client
        .post(url)
        .json(&openai::TokenizeRequest {
            model: model.to_string(),
            prompt,
        })
        .timeout(Duration::from_secs(30));
    if let Some(key) = api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
    }

    let response = match req_builder.send().await {
        Ok(resp) => resp,
        Err(err) => {
            tracing::warn!("upstream /tokenize unreachable ({url}): {err}; using local estimate");
            return None;
        }
    };
    if !response.status().is_success() {
        tracing::warn!(
            "upstream /tokenize returned {}; using local estimate",
            response.status()
        );
        return None;
    }
    let tokenized: openai::TokenizeResponse = match response.json().await {
        Ok(body) => body,
        Err(err) => {
            tracing::warn!("upstream /tokenize body unparseable: {err}; using local estimate");
            return None;
        }
    };

    // Exact content tokens + the chat template's per-message overhead.
    Some(tokenized.count + message_count as u32 * pipeline::PER_MESSAGE_OVERHEAD_TOKENS)
}

/// Compute a `max_tokens` that actually fits the context window after an overflow 400.
///
/// We clamp against the *larger* of two input estimates:
///   1. tokenizing the actual outgoing request (true size when the gateway can apply the
///      model's chat template — includes tool-injection scaffolding), and
///   2. the error's "at least N" figure, which equals `context + 1 - max_tokens` and is a
///      *hard lower bound* that tightens toward the real input as `max_tokens` shrinks.
///
/// Taking the max matters: a prompt-only tokenizer (gateway without `messages` support)
/// under-counts the tool template and would clamp *above* the real input, so the retry 400s
/// again and the re-clamp computes the same value — it never converges (field: 32000 → 15746
/// → fail, real input 89375 vs tokenized ~87272). The error's lower bound guarantees each
/// retry moves down. Returns `None` when the prompt alone fills the window (only compaction
/// can fix that).
async fn clamp_for_overflow(
    client: &Client,
    config: &Config,
    openai: &openai::OpenAIRequest,
    api_key: Option<&str>,
    message: &str,
) -> Option<u32> {
    let (context, error_input) = core::parse_context_overflow(message)?;

    let tokenized = match config.tokenize_urls().into_iter().next() {
        Some(url) => tokenize_openai_input(client, &url, openai, api_key).await,
        None => None,
    };
    let input = tokenized.unwrap_or(0).max(error_input);

    core::fit_output_to_window(context, input, openai.max_tokens)
}

/// Tokenize an outgoing chat request to get its true input size. Prefers the chat-aware
/// `/tokenize` (template applied); falls back to tokenizing the concatenated message/tool
/// text plus per-message overhead so it still works on gateways without `messages` support.
async fn tokenize_openai_input(
    client: &Client,
    url: &str,
    openai: &openai::OpenAIRequest,
    api_key: Option<&str>,
) -> Option<u32> {
    if let Some(count) = tokenize_chat(client, url, openai, api_key).await {
        return Some(count);
    }
    let (prompt, message_count) = openai_tokenize_text(openai);
    let mut req_builder = client
        .post(url)
        .json(&openai::TokenizeRequest {
            model: openai.model.clone(),
            prompt,
        })
        .timeout(Duration::from_secs(30));
    if let Some(key) = api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
    }
    let response = req_builder.send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let tokenized: openai::TokenizeResponse = response.json().await.ok()?;
    Some(tokenized.count + message_count as u32 * pipeline::PER_MESSAGE_OVERHEAD_TOKENS)
}

/// Flatten an OpenAI request to the text that contributes to the prompt (message content,
/// reasoning, tool-call arguments, and tool schemas) plus the message count. Base64 image
/// parts are skipped — they are not text and would massively over-count.
fn openai_tokenize_text(openai: &openai::OpenAIRequest) -> (String, usize) {
    let mut parts: Vec<String> = Vec::new();
    for m in &openai.messages {
        match &m.content {
            Some(openai::MessageContent::Text(t)) => parts.push(t.clone()),
            Some(openai::MessageContent::Parts(ps)) => {
                for p in ps {
                    if let openai::ContentPart::Text { text } = p {
                        parts.push(text.clone());
                    }
                }
            }
            None => {}
        }
        if let Some(rc) = &m.reasoning_content {
            parts.push(rc.clone());
        }
        if let Some(tcs) = &m.tool_calls {
            for tc in tcs {
                parts.push(tc.function.name.clone());
                parts.push(tc.function.arguments.clone());
            }
        }
    }
    if let Some(tools) = &openai.tools {
        for t in tools {
            parts.push(t.function.name.clone());
            if let Some(d) = &t.function.description {
                parts.push(d.clone());
            }
            parts.push(t.function.parameters.to_string());
        }
    }
    (parts.join("\n"), openai.messages.len())
}

pub async fn list_models_handler(
    Extension(config): Extension<Arc<Config>>,
    Extension(client): Extension<Client>,
    headers: HeaderMap,
) -> ProxyResult<Response> {
    let api_key = resolve_api_key(&config, &headers);
    let urls = config.models_urls();
    let mut last_err = None;

    for url in &urls {
        tracing::debug!("Fetching models from {}", url);

        let mut req_builder = client.get(url).timeout(Duration::from_secs(60));
        if let Some(ref key) = api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }

        match req_builder.send().await {
            Ok(response) if response.status().is_success() => {
                let openai_resp: openai::ModelsListResponse = response.json().await?;
                let anthropic_resp = pipeline::translate_models_list(openai_resp);
                return Ok(Json(anthropic_resp).into_response());
            }
            Ok(response) => {
                let status = response.status();
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                tracing::warn!("Upstream {} returned {}: {}", url, status, error_text);
                if is_retriable_status(status.as_u16()) {
                    last_err = Some(format!("Upstream returned {}: {}", status, error_text));
                    continue;
                }
                return Err(ProxyError::Upstream(format!(
                    "Upstream returned {}: {}",
                    status, error_text
                )));
            }
            Err(err) => {
                tracing::warn!("Failed to reach {}: {:?}", url, err);
                last_err = Some(format!("HTTP error: {}", err));
                continue;
            }
        }
    }

    Err(ProxyError::Upstream(
        last_err.unwrap_or_else(|| "All upstreams failed".to_string()),
    ))
}

fn resolve_api_key(config: &Config, headers: &HeaderMap) -> Option<String> {
    if config.passthrough_api_key {
        headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
    } else {
        config.api_key.clone()
    }
}

fn translation_policy(config: &Config) -> pipeline::TranslationPolicy {
    pipeline::TranslationPolicy {
        reasoning_model: config.reasoning_model.clone(),
        completion_model: config.completion_model.clone(),
        model_map: config.model_map.clone(),
        effort_map: config.effort_map.clone(),
        ignore_terms: config.system_prompt_ignore_terms.clone(),
    }
}

/// Max attempts per upstream URL (initial try + retries) for transient failures.
const MAX_ATTEMPTS: usize = 3;

/// Max context-overflow clamps per request. Each clamp tightens against the error's rising
/// lower bound, so a few passes converge even when the local tokenizer under-counts.
const MAX_CLAMP_RETRIES: u32 = 3;

fn is_retriable_status(status: u16) -> bool {
    matches!(status, 429 | 500..=599)
}

/// A transport-level reqwest error worth retrying with a fresh connection —
/// notably "connection closed before message completed" from a stale pooled
/// keep-alive socket, which is the dominant cause of intermittent 502s under load.
fn is_transient(err: &reqwest::Error) -> bool {
    err.is_connect() || err.is_timeout() || err.is_request() || err.is_body() || err.is_decode()
}

/// Extract a human-readable message from an upstream error body, preferring the
/// OpenAI `{"error":{"message":...}}` / `{"message":...}` shapes, falling back to
/// the raw (truncated) text.
fn upstream_message(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .or_else(|| v.get("message"))
                .and_then(|m| m.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| body.chars().take(500).collect())
}

/// Outcome of a single upstream send attempt.
enum SendOutcome {
    /// 2xx response, ready to consume.
    Ok(reqwest::Response),
    /// Transport error worth retrying with a fresh connection.
    Retriable(ProxyError),
    /// Upstream returned a non-2xx HTTP status (code preserved for the client).
    Status {
        status: StatusCode,
        message: String,
        retriable: bool,
    },
    /// Non-retriable transport error; surface immediately.
    Fatal(ProxyError),
}

/// Issue one POST to `url` and classify the result. On a 2xx the response body is
/// left unread so streaming and non-streaming callers can consume it differently.
async fn send_request(
    client: &Client,
    url: &str,
    openai_req: &openai::OpenAIRequest,
    api_key: Option<&str>,
) -> SendOutcome {
    let mut req_builder = client
        .post(url)
        .json(openai_req)
        .timeout(Duration::from_secs(600));

    if let Some(key) = api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
    }

    match req_builder.send().await {
        Ok(resp) if resp.status().is_success() => SendOutcome::Ok(resp),
        Ok(resp) => {
            let status = resp.status();
            let body = resp
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            SendOutcome::Status {
                status,
                message: upstream_message(&body),
                retriable: is_retriable_status(status.as_u16()),
            }
        }
        Err(err) => {
            if is_transient(&err) {
                SendOutcome::Retriable(ProxyError::Http(err))
            } else {
                SendOutcome::Fatal(ProxyError::Http(err))
            }
        }
    }
}

async fn handle_non_streaming(
    config: Arc<Config>,
    client: Client,
    mut openai_req: openai::OpenAIRequest,
    api_key: Option<String>,
) -> ProxyResult<Response> {
    let urls = config.chat_completions_urls();
    let mut last_err = None;
    let mut clamp_attempts = 0u32;

    for url in &urls {
        for attempt in 1..=MAX_ATTEMPTS {
            tracing::debug!(
                "Non-streaming request to {} (model: {}, attempt {}/{})",
                url,
                openai_req.model,
                attempt,
                MAX_ATTEMPTS
            );

            let upstream_start = Instant::now();
            let response = match send_request(&client, url, &openai_req, api_key.as_deref()).await {
                SendOutcome::Ok(resp) => {
                    metrics::upstream_latency(
                        upstream_start.elapsed().as_secs_f64(),
                        "chat_completions",
                    );
                    resp
                }
                SendOutcome::Retriable(err) => {
                    metrics::upstream_error("chat_completions");
                    tracing::warn!(
                        "Upstream {} attempt {}/{} failed: {}",
                        url,
                        attempt,
                        MAX_ATTEMPTS,
                        err
                    );
                    last_err = Some(err);
                    continue;
                }
                SendOutcome::Status {
                    status,
                    message,
                    retriable,
                } => {
                    metrics::upstream_error("chat_completions");
                    // Self-heal a context-length overflow once: clamp max_tokens so
                    // input + output fits the window, then retry. This unblocks the
                    // deadlock where even /compact can't run because it requests output.
                    if status.as_u16() == 400 && clamp_attempts < MAX_CLAMP_RETRIES {
                        if let Some(new_max) = clamp_for_overflow(
                            &client,
                            &config,
                            &openai_req,
                            api_key.as_deref(),
                            &message,
                        )
                        .await
                        {
                            tracing::warn!(
                                "upstream {url} context overflow; clamping max_tokens {:?} -> {new_max} and retrying",
                                openai_req.max_tokens
                            );
                            openai_req.max_tokens = Some(new_max);
                            clamp_attempts += 1;
                            continue;
                        }
                    }
                    if retriable {
                        tracing::warn!(
                            "upstream {url} attempt {attempt}/{MAX_ATTEMPTS} returned {status}: {message}"
                        );
                        last_err = Some(ProxyError::UpstreamStatus { status, message });
                        continue;
                    }
                    // 4xx is deterministic — surface the real status instead of masking as 502.
                    tracing::warn!("upstream {url} returned {status} (non-retriable): {message}");
                    return Err(ProxyError::UpstreamStatus { status, message });
                }
                SendOutcome::Fatal(err) => {
                    tracing::warn!("upstream {url} fatal transport error: {err}");
                    return Err(err);
                }
            };

            // Read the body explicitly so a mid-body transport drop is retried
            // rather than surfacing to the client as an unlogged 502.
            let bytes = match response.bytes().await {
                Ok(bytes) => bytes,
                Err(err) => {
                    metrics::upstream_error("chat_completions");
                    tracing::warn!(
                        "Upstream {} attempt {}/{} body read failed: {}",
                        url,
                        attempt,
                        MAX_ATTEMPTS,
                        err
                    );
                    last_err = Some(ProxyError::Http(err));
                    continue;
                }
            };

            let openai_resp: openai::OpenAIResponse = match serde_json::from_slice(&bytes) {
                Ok(resp) => resp,
                Err(err) => {
                    let preview: String =
                        String::from_utf8_lossy(&bytes).chars().take(500).collect();
                    tracing::error!(
                        "Failed to parse upstream response from {}: {} (body: {})",
                        url,
                        err,
                        preview
                    );
                    return Err(ProxyError::Upstream(format!(
                        "Invalid upstream response: {}",
                        err
                    )));
                }
            };

            metrics::tokens(
                openai_resp.usage.prompt_tokens,
                openai_resp.usage.completion_tokens,
                &openai_req.model,
            );

            if config.verbose {
                tracing::trace!(
                    "Received OpenAI response: {}",
                    serde_json::to_string_pretty(&openai_resp).unwrap_or_default()
                );
            }

            let anthropic_resp = pipeline::translate_response(openai_resp, &openai_req.model)?;

            if config.verbose {
                tracing::trace!(
                    "Transformed Anthropic response: {}",
                    serde_json::to_string_pretty(&anthropic_resp).unwrap_or_default()
                );
            }

            return Ok(Json(anthropic_resp).into_response());
        }
    }

    Err(last_err.unwrap_or_else(|| ProxyError::Upstream("All upstreams failed".to_string())))
}

async fn handle_streaming(
    config: Arc<Config>,
    client: Client,
    mut openai_req: openai::OpenAIRequest,
    api_key: Option<String>,
) -> ProxyResult<Response> {
    let urls = config.chat_completions_urls();
    let mut last_err = None;
    let mut clamp_attempts = 0u32;

    // Only the connection handshake is retried; once bytes start streaming we are
    // committed (events may already have reached the client).
    for url in &urls {
        for attempt in 1..=MAX_ATTEMPTS {
            tracing::debug!(
                "Streaming request to {} (model: {}, attempt {}/{})",
                url,
                openai_req.model,
                attempt,
                MAX_ATTEMPTS
            );

            let upstream_start = Instant::now();
            let response = match send_request(&client, url, &openai_req, api_key.as_deref()).await {
                SendOutcome::Ok(resp) => {
                    metrics::upstream_latency(
                        upstream_start.elapsed().as_secs_f64(),
                        "chat_completions",
                    );
                    resp
                }
                SendOutcome::Retriable(err) => {
                    metrics::upstream_error("chat_completions");
                    tracing::warn!(
                        "Upstream {} attempt {}/{} failed: {}",
                        url,
                        attempt,
                        MAX_ATTEMPTS,
                        err
                    );
                    last_err = Some(err);
                    continue;
                }
                SendOutcome::Status {
                    status,
                    message,
                    retriable,
                } => {
                    metrics::upstream_error("chat_completions");
                    // Self-heal a context-length overflow once: clamp max_tokens so
                    // input + output fits the window, then retry. This unblocks the
                    // deadlock where even /compact can't run because it requests output.
                    if status.as_u16() == 400 && clamp_attempts < MAX_CLAMP_RETRIES {
                        if let Some(new_max) = clamp_for_overflow(
                            &client,
                            &config,
                            &openai_req,
                            api_key.as_deref(),
                            &message,
                        )
                        .await
                        {
                            tracing::warn!(
                                "upstream {url} context overflow; clamping max_tokens {:?} -> {new_max} and retrying",
                                openai_req.max_tokens
                            );
                            openai_req.max_tokens = Some(new_max);
                            clamp_attempts += 1;
                            continue;
                        }
                    }
                    if retriable {
                        tracing::warn!(
                            "upstream {url} attempt {attempt}/{MAX_ATTEMPTS} returned {status}: {message}"
                        );
                        last_err = Some(ProxyError::UpstreamStatus { status, message });
                        continue;
                    }
                    // 4xx is deterministic — surface the real status instead of masking as 502.
                    tracing::warn!("upstream {url} returned {status} (non-retriable): {message}");
                    return Err(ProxyError::UpstreamStatus { status, message });
                }
                SendOutcome::Fatal(err) => {
                    tracing::warn!("upstream {url} fatal transport error: {err}");
                    return Err(err);
                }
            };

            let upstream = response.bytes_stream();
            let input_estimate = pipeline::estimate_openai_input_tokens(&openai_req);
            let sse_stream = create_sse_stream(
                upstream,
                openai_req.model.clone(),
                input_estimate,
                Duration::from_secs(config.heartbeat_secs),
            );

            let mut headers = HeaderMap::new();
            headers.insert(
                "Content-Type",
                HeaderValue::from_static("text/event-stream"),
            );
            headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
            headers.insert("Connection", HeaderValue::from_static("keep-alive"));

            return Ok((headers, Body::from_stream(sse_stream)).into_response());
        }
    }

    Err(last_err.unwrap_or_else(|| ProxyError::Upstream("All upstreams failed".to_string())))
}

fn serialize_event(event: &anthropic::StreamEvent) -> String {
    format!(
        "event: {}\ndata: {}\n\n",
        event.event_type(),
        serde_json::to_string(event).unwrap_or_default()
    )
}

/// Locate the next complete SSE frame boundary, returning `(offset, separator_len)`.
/// Handles both `\n\n` (the common case) and `\r\n\r\n` (CRLF upstreams), picking
/// whichever blank line appears first so neither framing style stalls the stream.
fn find_frame_boundary(buffer: &str) -> Option<(usize, usize)> {
    match (buffer.find("\n\n"), buffer.find("\r\n\r\n")) {
        (Some(lf), Some(crlf)) if lf <= crlf => Some((lf, 2)),
        (_, Some(crlf)) => Some((crlf, 4)),
        (Some(lf), None) => Some((lf, 2)),
        (None, None) => None,
    }
}

fn create_sse_stream(
    upstream: impl Stream<Item = Result<Bytes, impl std::fmt::Display + Send + 'static>>
        + Send
        + 'static,
    fallback_model: String,
    input_estimate: u32,
    heartbeat: Duration,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    // Heartbeat: emit an SSE comment (": keep-alive\n\n") after `heartbeat` of output
    // silence so a fronting proxy (Cloudflare free plan: 100s with no bytes → 524)
    // doesn't abort the response during the gap before the first token (the gateway's
    // queueing + prefill + initial thinking can exceed 100s).
    //
    // The timer is driven independently of upstream reads and reset only when we yield a
    // REAL event — deliberately. The Go gateway in front sends its OWN ": keep-alive"
    // comments, which this parser silently drops (see the heartbeat_comments test); if we
    // keyed the timer off upstream activity, those comments would keep resetting it and we
    // would never emit to our own client — 524ing anyway. Resetting only on real output
    // means a heartbeat fires whenever the *client* has gone quiet, regardless of upstream
    // chatter. A comment is transport-level (ignored by every SSE/Anthropic client) and is
    // only ever yielded between complete frames, so it can never corrupt an event.
    let period = if heartbeat.is_zero() {
        Duration::from_secs(60 * 60 * 24 * 365) // disabled → effectively never fires
    } else {
        heartbeat
    };
    async_stream::stream! {
        let mut buffer = String::new();
        let mut state = stream::initial_state(fallback_model, input_estimate);

        tokio::pin!(upstream);

        let mut beat = tokio::time::interval(period);
        beat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        beat.tick().await; // consume the immediate first tick → first beat is one period out

        loop {
            tokio::select! {
                maybe_chunk = upstream.next() => {
                    let Some(chunk) = maybe_chunk else { break };
                    match chunk {
                        Ok(bytes) => {
                            let mut yielded = false;
                            let text = String::from_utf8_lossy(&bytes);
                            buffer.push_str(&text);

                            while let Some((pos, sep_len)) = find_frame_boundary(&buffer) {
                                let line = buffer[..pos].to_string();
                                buffer = buffer[pos + sep_len..].to_string();

                                if line.trim().is_empty() {
                                    continue;
                                }

                                for l in line.lines() {
                                    if let Some(data) = l.strip_prefix("data: ") {
                                        if data.trim() == "[DONE]" {
                                            for event in stream::translate_done(&mut state) {
                                                yielded = true;
                                                yield Ok(Bytes::from(serialize_event(&event)));
                                            }
                                            continue;
                                        }

                                        if let Ok(chunk) = serde_json::from_str::<openai::StreamChunk>(data) {
                                            for event in stream::translate_chunk(&mut state, &chunk) {
                                                yielded = true;
                                                yield Ok(Bytes::from(serialize_event(&event)));
                                            }
                                        } else {
                                            tracing::debug!("Ignoring unrecognized upstream stream chunk: {}", data);
                                        }
                                    }
                                }
                            }

                            if yielded {
                                beat.reset(); // real output flowed → restart the idle timer
                            }
                        }
                        Err(e) => {
                            tracing::error!("Stream error: {}", e);
                            for event in stream::translate_error(format!("Stream error: {}", e)) {
                                yield Ok(Bytes::from(serialize_event(&event)));
                            }
                            break;
                        }
                    }
                }
                _ = beat.tick() => {
                    yield Ok(Bytes::from_static(b": keep-alive\n\n"));
                }
            }
        }

        // Safety net: if the upstream closed without a `[DONE]` marker, still emit the
        // closing message_delta (with usage) + message_stop. No-op if already finalized.
        for event in stream::translate_done(&mut state) {
            yield Ok(Bytes::from(serialize_event(&event)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{create_sse_stream, request_fields_summary};
    use crate::models::anthropic;
    use bytes::Bytes;
    use futures::stream::{self, StreamExt};
    use serde_json::{json, Value};
    use std::fmt;

    #[test]
    fn request_fields_summary_strips_content_keeps_unknown_fields() {
        let req: anthropic::AnthropicRequest = serde_json::from_value(json!({
            "model": "claude-opus-4-8",
            "max_tokens": 4096,
            "system": "you are helpful",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"}
            ],
            "tools": [
                {"name": "Bash", "input_schema": {}},
                {"type": "web_search_20250305", "name": "web_search"}
            ],
            // an unknown/new field we want to surface for debugging
            "output_config": {"effort": "xhigh"}
        }))
        .unwrap();

        let summary: Value = serde_json::from_str(&request_fields_summary(&req)).unwrap();
        // Bulky content replaced by counts.
        assert!(summary.get("messages").is_none());
        assert!(summary.get("system").is_none());
        assert_eq!(summary["messages_count"], json!(2));
        assert_eq!(summary["tools_count"], json!(2));
        assert_eq!(summary["system_blocks"], json!(1));
        // Tool identifiers are kept (type for server tools, name for custom) so a
        // server-side `web_search` request is visible in the debug log.
        assert_eq!(summary["tools"], json!(["Bash", "web_search_20250305"]));
        // Control + unknown fields survive for debugging.
        assert_eq!(summary["model"], json!("claude-opus-4-8"));
        assert_eq!(summary["output_config"]["effort"], json!("xhigh"));
    }

    #[derive(Debug)]
    struct TestError;
    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test error")
        }
    }

    fn openai_chunk(
        id: &str,
        model: &str,
        content: Option<&str>,
        finish_reason: Option<&str>,
    ) -> String {
        let mut delta = json!({});
        if let Some(c) = content {
            delta["content"] = json!(c);
        }
        let mut choice = json!({ "index": 0, "delta": delta });
        if let Some(fr) = finish_reason {
            choice["finish_reason"] = json!(fr);
        }
        let chunk = json!({
            "id": id,
            "model": model,
            "choices": [choice],
        });
        format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap())
    }

    fn openai_chunk_with_reasoning(id: &str, model: &str, reasoning: &str) -> String {
        let chunk = json!({
            "id": id,
            "model": model,
            "choices": [{ "index": 0, "delta": { "reasoning": reasoning } }],
        });
        format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap())
    }

    fn openai_chunk_with_reasoning_content(id: &str, model: &str, reasoning: &str) -> String {
        let chunk = json!({
            "id": id,
            "model": model,
            "choices": [{ "index": 0, "delta": { "reasoning_content": reasoning } }],
        });
        format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap())
    }

    fn openai_chunk_with_tool_call(
        id: &str,
        model: &str,
        tool_id: Option<&str>,
        name: Option<&str>,
        args: Option<&str>,
        finish_reason: Option<&str>,
    ) -> String {
        let mut tc = json!({ "index": 0 });
        if let Some(tid) = tool_id {
            tc["id"] = json!(tid);
            tc["type"] = json!("function");
        }
        let mut func = json!({});
        if let Some(n) = name {
            func["name"] = json!(n);
        }
        if let Some(a) = args {
            func["arguments"] = json!(a);
        }
        if !func.as_object().unwrap().is_empty() {
            tc["function"] = func;
        }
        let mut choice = json!({ "index": 0, "delta": { "tool_calls": [tc] } });
        if let Some(fr) = finish_reason {
            choice["finish_reason"] = json!(fr);
        }
        let chunk = json!({
            "id": id,
            "model": model,
            "choices": [choice],
        });
        format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap())
    }

    fn openai_done() -> String {
        "data: [DONE]\n\n".to_string()
    }

    fn make_stream(
        chunks: Vec<String>,
    ) -> impl futures::Stream<Item = Result<Bytes, TestError>> + Send + 'static {
        stream::iter(chunks.into_iter().map(|c| Ok(Bytes::from(c))))
    }

    async fn collect_events(chunks: Vec<String>, model: &str) -> Vec<Value> {
        let s = make_stream(chunks);
        let sse = create_sse_stream(
            s,
            model.to_string(),
            0,
            std::time::Duration::from_secs(3600),
        );
        tokio::pin!(sse);

        let mut events = Vec::new();
        while let Some(Ok(bytes)) = sse.next().await {
            let text = String::from_utf8_lossy(&bytes);
            for segment in text.split("\n\n").filter(|s| !s.is_empty()) {
                if let Some(data_line) = segment.lines().find(|l| l.starts_with("data: ")) {
                    let json_str = data_line.strip_prefix("data: ").unwrap();
                    if let Ok(v) = serde_json::from_str::<Value>(json_str) {
                        events.push(v);
                    }
                }
            }
        }
        events
    }

    use crate::config::Config;
    use axum::http::HeaderMap;

    fn make_x_api_key_header(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::HeaderName::from_static("x-api-key"),
            axum::http::HeaderValue::from_str(value).unwrap(),
        );
        headers
    }

    #[test]
    fn upstream_message_extracts_openai_error_shape() {
        assert_eq!(
            super::upstream_message(
                r#"{"error":{"message":"bad model","type":"invalid_request_error"}}"#
            ),
            "bad model"
        );
        assert_eq!(
            super::upstream_message(r#"{"message":"flat message"}"#),
            "flat message"
        );
        // Non-JSON or unrecognized shapes fall back to the raw text.
        assert_eq!(
            super::upstream_message("plain text error"),
            "plain text error"
        );
    }

    #[tokio::test]
    async fn resolve_api_key_passthrough_extracts_x_api_key() {
        let config = Config {
            passthrough_api_key: true,
            api_key: None,
            ..Default::default()
        };
        let headers = make_x_api_key_header("sk-my-test-key");
        let key = super::resolve_api_key(&config, &headers);
        assert_eq!(key, Some("sk-my-test-key".to_string()));
    }

    #[tokio::test]
    async fn resolve_api_key_passthrough_ignores_empty_header() {
        let config = Config {
            passthrough_api_key: true,
            api_key: None,
            ..Default::default()
        };
        // Empty header value returns None
        let key = super::resolve_api_key(&config, &HeaderMap::new());
        assert_eq!(key, None);

        // Explicitly empty value also returns None
        let headers = make_x_api_key_header("");
        let key = super::resolve_api_key(&config, &headers);
        assert_eq!(key, None);
    }

    #[tokio::test]
    async fn resolve_api_key_passthrough_returns_none_when_missing() {
        let config = Config {
            passthrough_api_key: true,
            api_key: None,
            ..Default::default()
        };
        let headers = HeaderMap::new();
        let key = super::resolve_api_key(&config, &headers);
        assert_eq!(key, None);
    }

    #[tokio::test]
    async fn resolve_api_key_static_key_when_passthrough_disabled() {
        let config = Config {
            passthrough_api_key: false,
            api_key: Some("sk-upstream".to_string()),
            ..Default::default()
        };
        // Even if x-api-key is present, static key wins when passthrough is off
        let headers = make_x_api_key_header("sk-ignored");
        let key = super::resolve_api_key(&config, &headers);
        assert_eq!(key, Some("sk-upstream".to_string()));
    }

    #[tokio::test]
    async fn resolve_api_key_both_missing_returns_none() {
        let config = Config {
            passthrough_api_key: false,
            api_key: None,
            ..Default::default()
        };
        let headers = HeaderMap::new();
        let key = super::resolve_api_key(&config, &headers);
        assert_eq!(key, None);
    }

    #[tokio::test]
    async fn text_stream_produces_message_start_content_block_and_stop() {
        let chunks = vec![
            openai_chunk("chatcmpl-1", "gpt-4o", Some("Hello"), None),
            openai_chunk("chatcmpl-1", "gpt-4o", Some(" world"), None),
            openai_chunk("chatcmpl-1", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];

        let events = collect_events(chunks, "fallback").await;

        assert_eq!(events[0]["type"], "message_start");
        assert_eq!(events[0]["message"]["id"], "chatcmpl-1");
        assert_eq!(events[0]["message"]["model"], "gpt-4o");
        assert_eq!(events[0]["message"]["role"], "assistant");

        assert_eq!(events[1]["type"], "content_block_start");
        assert_eq!(events[1]["content_block"]["type"], "text");

        assert_eq!(events[2]["type"], "content_block_delta");
        assert_eq!(events[2]["delta"]["type"], "text_delta");
        assert_eq!(events[2]["delta"]["text"], "Hello");

        assert_eq!(events[3]["type"], "content_block_delta");
        assert_eq!(events[3]["delta"]["text"], " world");

        assert_eq!(events[4]["type"], "content_block_stop");

        assert_eq!(events[5]["type"], "message_delta");
        assert_eq!(events[5]["delta"]["stop_reason"], "end_turn");

        assert_eq!(events[6]["type"], "message_stop");
    }

    #[tokio::test]
    async fn thinking_then_text_produces_two_content_blocks() {
        let chunks = vec![
            openai_chunk_with_reasoning("chatcmpl-2", "gpt-4o", "Let me think..."),
            openai_chunk_with_reasoning("chatcmpl-2", "gpt-4o", " more thinking"),
            openai_chunk("chatcmpl-2", "gpt-4o", Some("The answer is 42"), None),
            openai_chunk("chatcmpl-2", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];

        let events = collect_events(chunks, "fallback").await;

        assert_eq!(events[0]["type"], "message_start");
        assert_eq!(events[1]["type"], "content_block_start");
        assert_eq!(events[1]["content_block"]["type"], "thinking");
        assert_eq!(events[1]["index"], 0);
        assert_eq!(events[4]["type"], "content_block_stop");
        assert_eq!(events[4]["index"], 0);
        assert_eq!(events[5]["type"], "content_block_start");
        assert_eq!(events[5]["content_block"]["type"], "text");
        assert_eq!(events[5]["index"], 1);
    }

    #[tokio::test]
    async fn reasoning_content_stream_produces_thinking_block() {
        let chunks = vec![
            openai_chunk_with_reasoning_content("chatcmpl-2", "gpt-4o", "Let me think..."),
            openai_chunk("chatcmpl-2", "gpt-4o", Some("The answer is 42"), None),
            openai_chunk("chatcmpl-2", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];

        let events = collect_events(chunks, "fallback").await;

        assert_eq!(events[1]["type"], "content_block_start");
        assert_eq!(events[1]["content_block"]["type"], "thinking");
        assert_eq!(events[2]["delta"]["type"], "thinking_delta");
        assert_eq!(events[2]["delta"]["thinking"], "Let me think...");
    }

    #[tokio::test]
    async fn tool_call_stream_produces_tool_use_block() {
        let chunks = vec![
            openai_chunk_with_tool_call(
                "chatcmpl-3",
                "gpt-4o",
                Some("call_abc"),
                Some("read_file"),
                None,
                None,
            ),
            openai_chunk_with_tool_call(
                "chatcmpl-3",
                "gpt-4o",
                None,
                None,
                Some("{\"path\":"),
                None,
            ),
            openai_chunk_with_tool_call(
                "chatcmpl-3",
                "gpt-4o",
                None,
                None,
                Some("\"/tmp\"}"),
                None,
            ),
            openai_chunk("chatcmpl-3", "gpt-4o", None, Some("tool_calls")),
            openai_done(),
        ];

        let events = collect_events(chunks, "fallback").await;
        assert_eq!(events[1]["content_block"]["type"], "tool_use");
        assert_eq!(events[1]["content_block"]["id"], "call_abc");
        assert_eq!(events[5]["delta"]["stop_reason"], "tool_use");
    }

    #[tokio::test]
    async fn done_without_finish_reason_still_produces_message_stop() {
        let chunks = vec![
            openai_chunk("chatcmpl-4", "gpt-4o", Some("hi"), None),
            openai_done(),
        ];
        let events = collect_events(chunks, "fallback").await;
        assert_eq!(events.last().unwrap()["type"], "message_stop");
    }

    #[tokio::test]
    async fn fallback_model_used_when_upstream_omits_model() {
        let chunk = json!({
            "choices": [{ "index": 0, "delta": { "content": "hey" } }],
        });
        let chunks = vec![
            format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap()),
            openai_chunk("id", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];
        let events = collect_events(chunks, "my-fallback-model").await;
        assert_eq!(events[0]["message"]["model"], "my-fallback-model");
    }

    #[tokio::test]
    async fn empty_content_chunks_are_not_emitted() {
        let chunks = vec![
            openai_chunk("chatcmpl-5", "gpt-4o", Some(""), None),
            openai_chunk("chatcmpl-5", "gpt-4o", Some("hello"), None),
            openai_chunk("chatcmpl-5", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];
        let events = collect_events(chunks, "fallback").await;
        let text_deltas: Vec<_> = events
            .iter()
            .filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta")
            .collect();
        assert_eq!(text_deltas.len(), 1);
        assert_eq!(text_deltas[0]["delta"]["text"], "hello");
    }

    #[tokio::test]
    async fn stream_error_produces_error_event_and_stops() {
        let items: Vec<Result<Bytes, TestError>> = vec![
            Ok(Bytes::from(openai_chunk(
                "chatcmpl-6",
                "gpt-4o",
                Some("start"),
                None,
            ))),
            Err(TestError),
        ];
        let s = stream::iter(items);
        let sse = create_sse_stream(
            s,
            "fallback".to_string(),
            0,
            std::time::Duration::from_secs(3600),
        );
        tokio::pin!(sse);

        let mut events = Vec::new();
        while let Some(Ok(bytes)) = sse.next().await {
            let text = String::from_utf8_lossy(&bytes);
            for segment in text.split("\n\n").filter(|s| !s.is_empty()) {
                if let Some(data_line) = segment.lines().find(|l| l.starts_with("data: ")) {
                    let json_str = data_line.strip_prefix("data: ").unwrap();
                    if let Ok(v) = serde_json::from_str::<Value>(json_str) {
                        events.push(v);
                    }
                }
            }
        }
        let error_events: Vec<_> = events.iter().filter(|e| e["type"] == "error").collect();
        assert_eq!(error_events.len(), 1);
    }

    #[tokio::test]
    async fn chunked_delivery_handles_split_sse_frames() {
        let full_chunk = openai_chunk("chatcmpl-7", "gpt-4o", Some("split"), None);
        let mid = full_chunk.len() / 2;
        let part1 = full_chunk[..mid].to_string();
        let part2 = format!(
            "{}{}{}",
            &full_chunk[mid..],
            openai_chunk("chatcmpl-7", "gpt-4o", None, Some("stop")),
            openai_done()
        );
        let events = collect_events(vec![part1, part2], "fallback").await;
        let text_deltas: Vec<_> = events
            .iter()
            .filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta")
            .collect();
        assert_eq!(text_deltas.len(), 1);
        assert_eq!(text_deltas[0]["delta"]["text"], "split");
    }

    #[tokio::test]
    async fn crlf_framed_stream_is_parsed() {
        // Some upstreams terminate SSE frames with CRLF (\r\n\r\n).
        let frame = |body: &str| {
            let chunk = json!({
                "id": "chatcmpl-crlf", "model": "gpt-4o",
                "choices": [{ "index": 0, "delta": { "content": body } }],
            });
            format!("data: {}\r\n\r\n", serde_json::to_string(&chunk).unwrap())
        };
        let chunks = vec![frame("CR"), frame("LF"), "data: [DONE]\r\n\r\n".to_string()];
        let events = collect_events(chunks, "fallback").await;
        let text: String = events
            .iter()
            .filter(|e| e["delta"]["type"] == "text_delta")
            .map(|e| e["delta"]["text"].as_str().unwrap_or_default())
            .collect();
        assert_eq!(text, "CRLF");
        assert_eq!(events.last().unwrap()["type"], "message_stop");
    }

    #[tokio::test]
    async fn heartbeat_comments_and_blank_lines_are_ignored() {
        let chunks = vec![
            ": keep-alive\n\n".to_string(),
            "\n\n".to_string(),
            openai_chunk("chatcmpl-hb", "gpt-4o", Some("hello"), None),
            ": ping\n\n".to_string(),
            openai_chunk("chatcmpl-hb", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];
        let events = collect_events(chunks, "fallback").await;
        let text_deltas: Vec<_> = events
            .iter()
            .filter(|e| e["delta"]["type"] == "text_delta")
            .collect();
        assert_eq!(text_deltas.len(), 1);
        assert_eq!(text_deltas[0]["delta"]["text"], "hello");
    }

    // A backend that stalls before the first token must not leave our client silent:
    // keep-alive comments should fill the gap, the real events should still arrive
    // intact, and the comments must sit strictly before the first translated event.
    #[tokio::test]
    async fn heartbeat_fills_pre_token_gap_then_real_events_flow() {
        use std::time::Duration;
        let upstream = async_stream::stream! {
            tokio::time::sleep(Duration::from_millis(200)).await; // ~6 beats at 30ms
            yield Ok::<_, TestError>(Bytes::from(openai_chunk("c", "gpt-4o", Some("hi"), None)));
            yield Ok(Bytes::from(openai_chunk("c", "gpt-4o", None, Some("stop"))));
            yield Ok(Bytes::from(openai_done()));
        };
        let sse = create_sse_stream(
            upstream,
            "fallback".to_string(),
            0,
            Duration::from_millis(30),
        );
        tokio::pin!(sse);

        let mut raw = String::new();
        while let Some(Ok(bytes)) = sse.next().await {
            raw.push_str(&String::from_utf8_lossy(&bytes));
        }

        // Heartbeat(s) fired during the stall, and the real Anthropic events are intact.
        assert!(
            raw.contains(": keep-alive\n\n"),
            "no heartbeat emitted: {raw:?}"
        );
        assert!(
            raw.contains("event: message_start"),
            "missing message_start: {raw:?}"
        );
        assert!(
            raw.contains("event: message_stop"),
            "missing message_stop: {raw:?}"
        );

        // Every keep-alive precedes the first real event — never spliced into a frame.
        let first_event = raw.find("event: ").expect("an event exists");
        let last_hb = raw.rfind(": keep-alive").expect("a heartbeat exists");
        assert!(
            last_hb < first_event,
            "heartbeat appeared at/after the first event (mid-stream injection): {raw:?}"
        );
    }

    #[tokio::test]
    async fn malformed_json_chunk_is_skipped_without_aborting() {
        let chunks = vec![
            openai_chunk("chatcmpl-m", "gpt-4o", Some("good"), None),
            "data: {not valid json}\n\n".to_string(),
            openai_chunk("chatcmpl-m", "gpt-4o", Some(" still going"), None),
            openai_chunk("chatcmpl-m", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];
        let events = collect_events(chunks, "fallback").await;
        let text: String = events
            .iter()
            .filter(|e| e["delta"]["type"] == "text_delta")
            .map(|e| e["delta"]["text"].as_str().unwrap_or_default())
            .collect();
        assert_eq!(text, "good still going");
        assert_eq!(events.last().unwrap()["type"], "message_stop");
    }

    #[tokio::test]
    async fn stream_chunk_with_partial_usage_does_not_abort() {
        // A finish chunk that carries usage with only some fields must still parse.
        let finish = json!({
            "id": "chatcmpl-u", "model": "gpt-4o",
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 9 }
        });
        let chunks = vec![
            openai_chunk("chatcmpl-u", "gpt-4o", Some("hi"), None),
            format!("data: {}\n\n", serde_json::to_string(&finish).unwrap()),
            openai_done(),
        ];
        let events = collect_events(chunks, "fallback").await;
        let delta = events
            .iter()
            .find(|e| e["type"] == "message_delta")
            .expect("message_delta present");
        assert_eq!(delta["usage"]["input_tokens"], 9);
        assert_eq!(delta["usage"]["output_tokens"], 0);
    }

    #[tokio::test]
    async fn text_then_tool_call_produces_two_blocks() {
        let chunks = vec![
            openai_chunk("chatcmpl-8", "gpt-4o", Some("I'll read that file."), None),
            openai_chunk_with_tool_call(
                "chatcmpl-8",
                "gpt-4o",
                Some("call_xyz"),
                Some("read_file"),
                None,
                None,
            ),
            openai_chunk_with_tool_call(
                "chatcmpl-8",
                "gpt-4o",
                None,
                None,
                Some("{\"path\":\"/etc\"}"),
                None,
            ),
            openai_chunk("chatcmpl-8", "gpt-4o", None, Some("tool_calls")),
            openai_done(),
        ];
        let events = collect_events(chunks, "fallback").await;
        let block_starts: Vec<_> = events
            .iter()
            .filter(|e| e["type"] == "content_block_start")
            .collect();
        assert_eq!(block_starts.len(), 2);
        assert_eq!(block_starts[0]["content_block"]["type"], "text");
        assert_eq!(block_starts[1]["content_block"]["type"], "tool_use");
    }
}
