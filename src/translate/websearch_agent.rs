//! Emulates Anthropic's server-side `web_search` / `web_fetch` tools for local models that
//! can't browse, by running the tool loop ourselves against the co-located `open-websearch`
//! server (see [`crate::websearch`]).
//!
//! Flow (see proxy.rs for where this is dispatched):
//!   1. The Anthropic request carries a `web_search_*` / `web_fetch_*` server tool. We rewrite
//!      it into a plain callable function tool (`web_search(query)` / `web_fetch(url)`) so the
//!      backend model can invoke it.
//!   2. We drive a non-streaming loop: call the backend → if it calls a web tool, run the real
//!      search/fetch and feed the result back → repeat until the model answers (or calls a
//!      client-side tool, which we hand back untouched).
//!   3. We assemble the faithful Anthropic content blocks — `server_tool_use` +
//!      `web_search_tool_result` / `web_fetch_tool_result` + `text` — and return them as a
//!      normal message (non-streaming) or replay them as SSE (streaming).
//!
//! Streaming keeps the connection alive with `: keep-alive` heartbeats while the loop runs, so
//! a long multi-round search doesn't trip the fronting proxy's idle timeout (Cloudflare 100s).

use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::models::{anthropic, openai};
use crate::websearch::{FetchResult, SearchResult, WebSearchClient};
use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures::Stream;
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Per-call tuning. Engines/mode are fixed to the reliable DuckDuckGo + request combination
/// (Playwright isn't bundled); see docker-entrypoint.sh.
const SEARCH_LIMIT: u32 = 5;
const SEARCH_ENGINE: &str = "duckduckgo";
const SEARCH_MODE: &str = "request";
const FETCH_MAX_CHARS: u32 = 30000;
/// Safety bounds on how many tool calls we honor per request (mirrors Anthropic's `max_uses`).
const SEARCH_MAX_USES: u32 = 5;
const FETCH_MAX_USES: u32 = 10;
/// Combined search+fetch budget. Once reached, the next round is sent WITHOUT tools so the
/// model must answer with what it has, instead of endlessly reformulating queries (a small
/// model on a vague prompt like "today's news" will otherwise loop until MAX_ROUNDS). Kept low
/// because each round costs a backend round-trip — most queries need 1-2 searches.
const MAX_WEB_USES: u32 = 3;
/// Hard cap on backend round-trips, so a model that loops forever can't pin a worker. The last
/// round is always toolless (forced answer), so the loop always terminates with text.
const MAX_ROUNDS: usize = 5;
/// Attempts per backend call. We fire immediately (no client-side queueing) and only retry on a
/// 5xx (or transport error) — a busy backend the gateway reports as overloaded — never on a 4xx.
const BACKEND_MAX_ATTEMPTS: u32 = 3;
/// Fallback search-agent model if `ANTHROPIC_PROXY_WEBSEARCH_MODEL` somehow isn't set when we
/// reach the loop. Normally the configured model (e.g. `auto`) is used, so the agent's
/// backend calls load-balance across healthy backends rather than pinning the client's model.
/// The concrete model the balancer picks is read back from the response and reported.
const AGENT_MODEL: &str = "auto";
/// Cap each backend round's output. Clients (e.g. Cowork) send `max_tokens: 64000`, which on a
/// local model makes every reasoning round crawl; a search answer never needs that much, and
/// capping keeps each round fast so the loop can't stall for minutes.
const AGENT_MAX_TOKENS: u32 = 8192;
/// Force a low reasoning effort for the agent's backend rounds. The client's effort (often
/// `high`) makes a small local model spend a minute+ "thinking" just to decide to search; the
/// loop is mechanical (decide → search → summarize) and doesn't need deep reasoning, so we
/// trade thinking depth for responsiveness here.
const AGENT_EFFORT: &str = "low";

/// Monotonic id source for the `srvtoolu_…` / `msg_…` ids we synthesize. Uniqueness within a
/// single message is all a client needs to match a result to its `server_tool_use`.
static ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_id(prefix: &str) -> String {
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{n:020x}")
}

fn is_search_type(t: &str) -> bool {
    t.starts_with("web_search")
}
fn is_fetch_type(t: &str) -> bool {
    t.starts_with("web_fetch")
}

/// Whether the request carries an Anthropic server browsing tool (`web_search*` / `web_fetch*`).
/// When false, the caller falls through to the normal proxy path.
pub fn detect(req: &anthropic::AnthropicRequest) -> bool {
    req.tools.as_ref().is_some_and(|tools| {
        tools.iter().any(|t| {
            t.tool_type
                .as_deref()
                .is_some_and(|ty| is_search_type(ty) || is_fetch_type(ty))
        })
    })
}

/// Remove the server browsing tools entirely (used when emulation is disabled): the model
/// then answers without web access, and the client gets no search results.
pub fn strip_web_tools(req: &mut anthropic::AnthropicRequest) {
    if let Some(tools) = req.tools.as_mut() {
        tools.retain(|t| {
            t.tool_type
                .as_deref()
                .map(|ty| !is_search_type(ty) && !is_fetch_type(ty))
                .unwrap_or(true)
        });
        if tools.is_empty() {
            req.tools = None;
        }
    }
}

/// Rewrite the server browsing tools in-place into plain callable function tools, so the
/// backend model can actually invoke them. Other (client-side) tools are left untouched.
pub fn rewrite_tools(req: &mut anthropic::AnthropicRequest) {
    let Some(tools) = req.tools.as_mut() else {
        return;
    };
    for t in tools.iter_mut() {
        let Some(ty) = t.tool_type.clone() else {
            continue;
        };
        if is_search_type(&ty) {
            t.tool_type = None;
            t.name = "web_search".to_string();
            t.description =
                Some("Search the web and return relevant results for a query.".to_string());
            t.input_schema = json!({
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query."}},
                "required": ["query"]
            });
        } else if is_fetch_type(&ty) {
            t.tool_type = None;
            t.name = "web_fetch".to_string();
            t.description = Some("Fetch and read the full text content of a URL.".to_string());
            t.input_schema = json!({
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL to fetch."}},
                "required": ["url"]
            });
        }
    }
}

/// The assembled outcome of the agent loop, ready to render either as a message or SSE.
struct AgentOutput {
    model: String,
    blocks: Vec<Value>,
    stop_reason: String,
    input_tokens: u32,
    output_tokens: u32,
    search_requests: u32,
    fetch_requests: u32,
}

impl AgentOutput {
    fn usage_value(&self) -> Value {
        let mut usage = json!({
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        });
        if self.search_requests > 0 || self.fetch_requests > 0 {
            let mut stu = serde_json::Map::new();
            if self.search_requests > 0 {
                stu.insert("web_search_requests".into(), json!(self.search_requests));
            }
            if self.fetch_requests > 0 {
                stu.insert("web_fetch_requests".into(), json!(self.fetch_requests));
            }
            usage["server_tool_use"] = Value::Object(stu);
        }
        usage
    }
}

/// Entry point. `openai_req` is the already-translated request (with the web tools rewritten
/// to function tools); `streaming` is what the client asked for.
pub async fn handle(
    config: Arc<Config>,
    client: Client,
    openai_req: openai::OpenAIRequest,
    api_key: Option<String>,
    streaming: bool,
) -> ProxyResult<Response> {
    if streaming {
        let period = if config.heartbeat_secs == 0 {
            Duration::from_secs(60 * 60 * 24 * 365)
        } else {
            Duration::from_secs(config.heartbeat_secs)
        };
        let model = openai_req.model.clone();
        let stream = sse_stream(config, client, openai_req, api_key, model, period);

        let mut headers = HeaderMap::new();
        headers.insert(
            "Content-Type",
            HeaderValue::from_static("text/event-stream"),
        );
        headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
        headers.insert("Connection", HeaderValue::from_static("keep-alive"));
        Ok((headers, Body::from_stream(stream)).into_response())
    } else {
        let out = run_loop(&config, &client, openai_req, api_key.as_deref()).await?;
        Ok(Json(json!({
            "id": next_id("msg"),
            "type": "message",
            "role": "assistant",
            "model": out.model,
            "content": out.blocks,
            "stop_reason": out.stop_reason,
            "stop_sequence": null,
            "usage": out.usage_value(),
        }))
        .into_response())
    }
}

/// Build the SSE stream: emit `message_start`, beat keep-alives while the loop runs, then
/// replay the assembled blocks and close with `message_delta` + `message_stop`.
fn sse_stream(
    config: Arc<Config>,
    client: Client,
    openai_req: openai::OpenAIRequest,
    api_key: Option<String>,
    model: String,
    period: Duration,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        // Announce the message immediately so the client sees a live response while we work.
        let msg_id = next_id("msg");
        yield Ok(Bytes::from(sse_event("message_start", &json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        }))));

        let fut = run_loop(&config, &client, openai_req, api_key.as_deref());
        tokio::pin!(fut);

        let mut beat = tokio::time::interval(period);
        beat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        beat.tick().await; // consume immediate tick → first beat is one period out

        let out = loop {
            tokio::select! {
                result = &mut fut => break result,
                _ = beat.tick() => { yield Ok(Bytes::from_static(b": keep-alive\n\n")); }
            }
        };

        match out {
            Ok(out) => {
                for frame in render_block_events(&out.blocks) {
                    yield Ok(Bytes::from(frame));
                }
                yield Ok(Bytes::from(sse_event("message_delta", &json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": out.stop_reason, "stop_sequence": null},
                    "usage": out.usage_value(),
                }))));
                yield Ok(Bytes::from(sse_event("message_stop", &json!({"type": "message_stop"}))));
            }
            Err(err) => {
                yield Ok(Bytes::from(sse_event("error", &json!({
                    "type": "error",
                    "error": {"type": "api_error", "message": err.to_string()},
                }))));
            }
        }
    }
}

/// Drive the model↔tool loop to completion and assemble the Anthropic content blocks.
async fn run_loop(
    config: &Config,
    client: &Client,
    mut openai_req: openai::OpenAIRequest,
    api_key: Option<&str>,
) -> ProxyResult<AgentOutput> {
    // Route every backend round-trip through the configured search-agent model (e.g. `auto`),
    // cap output so a client's huge max_tokens can't make each round crawl, and force low effort
    // so the model doesn't burn a minute thinking before each search.
    let agent_model = config
        .websearch_model
        .clone()
        .unwrap_or_else(|| AGENT_MODEL.to_string());
    openai_req.model = agent_model.clone();
    openai_req.max_tokens = Some(
        openai_req
            .max_tokens
            .map_or(AGENT_MAX_TOKENS, |m| m.min(AGENT_MAX_TOKENS)),
    );
    openai_req.reasoning_effort = Some(AGENT_EFFORT.to_string());

    let ws = WebSearchClient::new(config.websearch_url.clone(), client.clone());
    let mut messages = openai_req.messages.clone();
    // The loop is mechanical (decide → search → summarize); chain-of-thought just adds large
    // latency (a toolless answer round was observed generating 4.6K think tokens / 85s). Disable
    // it via the qwen `/no_think` soft switch on the system prompt.
    disable_thinking(&mut messages);
    let mut blocks: Vec<Value> = Vec::new();
    let mut out = AgentOutput {
        model: agent_model,
        blocks: Vec::new(),
        stop_reason: "end_turn".to_string(),
        input_tokens: 0,
        output_tokens: 0,
        search_requests: 0,
        fetch_requests: 0,
    };
    tracing::info!(max_rounds = MAX_ROUNDS, "web agent: start");

    for round in 0..MAX_ROUNDS {
        let mut req = openai_req.clone();
        req.messages = messages.clone();
        req.stream = Some(false);
        req.stream_options = None;

        // Once the search/fetch budget is spent (or on the final round), drop the tools and tell
        // the model to answer with what it has — otherwise a small model that still "wants" to
        // search will leak raw tool-call syntax as text instead of answering.
        let force_answer =
            round + 1 >= MAX_ROUNDS || out.search_requests + out.fetch_requests >= MAX_WEB_USES;
        if force_answer {
            req.tools = None;
            req.tool_choice = None;
            req.parallel_tool_calls = None;
            req.messages.push(openai::Message {
                role: "user".to_string(),
                content: Some(openai::MessageContent::Text(
                    "Stop searching. Using only the information already gathered above, write the \
                     final answer to my original request now. Do not call any tools or output any \
                     tool-call syntax."
                        .to_string(),
                )),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }

        let t0 = Instant::now();
        let resp = backend_call(client, config, &req, api_key).await?;
        // Report the concrete backend the balancer chose, not "auto".
        if let Some(model) = &resp.model {
            out.model = model.clone();
        }
        if round == 0 {
            out.input_tokens = resp.usage.prompt_tokens;
        }
        out.output_tokens += resp.usage.completion_tokens;

        let choice = resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProxyError::Upstream("upstream returned no choices".into()))?;
        let msg = choice.message;
        let tool_calls = msg.tool_calls.clone().unwrap_or_default();
        tracing::info!(
            round,
            backend = %out.model,
            ms = t0.elapsed().as_millis() as u64,
            tool_calls = tool_calls.len(),
            completion_tokens = resp.usage.completion_tokens,
            force_answer,
            "web agent: backend responded"
        );
        let all_web = !tool_calls.is_empty()
            && tool_calls
                .iter()
                .all(|tc| web_kind(&tc.function.name).is_some());

        // Terminal: a plain answer, or a client-side tool call we must hand back unexecuted.
        if !all_web {
            push_text(&mut blocks, msg.content.as_deref());
            if tool_calls.is_empty() {
                out.stop_reason = "end_turn".to_string();
            } else {
                for tc in &tool_calls {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": parse_args(&tc.function.arguments),
                    }));
                }
                out.stop_reason = "tool_use".to_string();
            }
            tracing::info!(
                rounds = round + 1,
                stop_reason = %out.stop_reason,
                blocks = blocks.len(),
                searches = out.search_requests,
                fetches = out.fetch_requests,
                "web agent: done"
            );
            out.blocks = blocks;
            return Ok(out);
        }

        // The model wants to browse. Record any decision text, keep the assistant turn in the
        // conversation, then run each web call and feed its result back.
        push_text(&mut blocks, msg.content.as_deref());
        messages.push(openai::Message {
            role: "assistant".to_string(),
            content: msg.content.clone().map(openai::MessageContent::Text),
            reasoning_content: None,
            tool_calls: Some(tool_calls.clone()),
            tool_call_id: None,
            name: None,
        });

        // Plan each call up front (assign ids, count budget, emit the server_tool_use block) in
        // order, then run the actual searches/fetches concurrently so a round with parallel tool
        // calls doesn't serialize.
        let mut plans: Vec<Plan> = Vec::new();
        for tc in &tool_calls {
            let args = parse_args(&tc.function.arguments);
            let srv_id = next_id("srvtoolu");
            let kind = web_kind(&tc.function.name).expect("all_web checked");
            let (arg, over_budget) = match kind {
                WebKind::Search => {
                    let query = args
                        .get("query")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    blocks.push(json!({
                        "type": "server_tool_use", "id": srv_id,
                        "name": "web_search", "input": {"query": query},
                    }));
                    out.search_requests += 1;
                    (query, out.search_requests > SEARCH_MAX_USES)
                }
                WebKind::Fetch => {
                    let url = args
                        .get("url")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    blocks.push(json!({
                        "type": "server_tool_use", "id": srv_id,
                        "name": "web_fetch", "input": {"url": url},
                    }));
                    out.fetch_requests += 1;
                    (url, out.fetch_requests > FETCH_MAX_USES)
                }
            };
            plans.push(Plan {
                tc_id: tc.id.clone(),
                srv_id,
                kind,
                arg,
                over_budget,
            });
        }

        // Run the (non-over-budget) calls concurrently.
        let executed = futures::future::join_all(plans.iter().map(|p| {
            let ws = &ws;
            async move {
                if p.over_budget {
                    return None;
                }
                let t = Instant::now();
                Some(match p.kind {
                    WebKind::Search => {
                        let engines = [SEARCH_ENGINE.to_string()];
                        CallResult::Search(
                            ws.search(&p.arg, SEARCH_LIMIT, &engines, SEARCH_MODE).await,
                            t.elapsed(),
                        )
                    }
                    WebKind::Fetch => {
                        CallResult::Fetch(ws.fetch(&p.arg, FETCH_MAX_CHARS).await, t.elapsed())
                    }
                })
            }
        }))
        .await;

        // Assemble result blocks + tool-result messages back in the original order.
        for (p, res) in plans.iter().zip(executed) {
            match res {
                None => {
                    let err_block = match p.kind {
                        WebKind::Search => search_error(&p.srv_id, "max_uses_exceeded"),
                        WebKind::Fetch => fetch_error(&p.srv_id, "max_uses_exceeded"),
                    };
                    blocks.push(err_block);
                    messages.push(tool_result_msg(&p.tc_id, "Tool use limit reached."));
                }
                Some(CallResult::Search(result, elapsed)) => match result {
                    Ok(results) => {
                        tracing::info!(
                            round, query = %p.arg, results = results.len(),
                            ms = elapsed.as_millis() as u64, "web agent: web_search ok"
                        );
                        let items: Vec<Value> = results
                            .iter()
                            .map(|r| json!({"type": "web_search_result", "url": r.url, "title": r.title}))
                            .collect();
                        blocks.push(json!({
                            "type": "web_search_tool_result",
                            "tool_use_id": p.srv_id,
                            "content": items,
                        }));
                        messages.push(tool_result_msg(&p.tc_id, &format_search(&results)));
                    }
                    Err(err) => {
                        tracing::warn!(
                            round, query = %p.arg, ms = elapsed.as_millis() as u64,
                            "web agent: web_search failed: {err}"
                        );
                        blocks.push(search_error(&p.srv_id, "unavailable"));
                        messages.push(tool_result_msg(&p.tc_id, &format!("Search failed: {err}")));
                    }
                },
                Some(CallResult::Fetch(result, elapsed)) => match result {
                    Ok(f) => {
                        let final_url = if f.final_url.is_empty() {
                            p.arg.clone()
                        } else {
                            f.final_url.clone()
                        };
                        tracing::info!(
                            round, url = %final_url, chars = f.content.len(),
                            ms = elapsed.as_millis() as u64, "web agent: web_fetch ok"
                        );
                        blocks.push(json!({
                            "type": "web_fetch_tool_result",
                            "tool_use_id": p.srv_id,
                            "content": {
                                "type": "web_fetch_result",
                                "url": final_url,
                                "content": {
                                    "type": "document",
                                    "source": {"type": "text", "media_type": "text/plain", "data": f.content},
                                    "title": f.title,
                                },
                            },
                        }));
                        messages.push(tool_result_msg(
                            &p.tc_id,
                            &format!("Fetched {}:\n{}", final_url, f.content),
                        ));
                    }
                    Err(err) => {
                        tracing::warn!(
                            round, url = %p.arg, ms = elapsed.as_millis() as u64,
                            "web agent: web_fetch failed: {err}"
                        );
                        blocks.push(fetch_error(&p.srv_id, "url_not_accessible"));
                        messages.push(tool_result_msg(&p.tc_id, &format!("Fetch failed: {err}")));
                    }
                },
            }
        }
        // Out of rounds: stop cleanly with whatever we have rather than looping forever.
        if round == MAX_ROUNDS - 1 {
            tracing::warn!("web tool loop hit MAX_ROUNDS; returning partial result");
        }
    }

    out.blocks = blocks;
    Ok(out)
}

#[derive(Clone, Copy)]
enum WebKind {
    Search,
    Fetch,
}

/// A single planned web call: ids/budget decided up front (ordered), the slow search/fetch run
/// concurrently afterwards.
struct Plan {
    tc_id: String,
    srv_id: String,
    kind: WebKind,
    arg: String,
    over_budget: bool,
}

/// The outcome of one concurrently-executed web call, carrying its own elapsed time for logging.
enum CallResult {
    Search(anyhow::Result<Vec<SearchResult>>, Duration),
    Fetch(anyhow::Result<FetchResult>, Duration),
}

fn web_kind(name: &str) -> Option<WebKind> {
    match name {
        "web_search" => Some(WebKind::Search),
        "web_fetch" => Some(WebKind::Fetch),
        _ => None,
    }
}

fn parse_args(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({}))
}

/// Append the qwen `/no_think` soft switch to the system prompt (or prepend a system message if
/// there isn't a text one), disabling chain-of-thought for the whole agent loop.
fn disable_thinking(messages: &mut Vec<openai::Message>) {
    for m in messages.iter_mut() {
        if m.role == "system" {
            if let Some(openai::MessageContent::Text(t)) = &mut m.content {
                if !t.contains("/no_think") {
                    t.push_str("\n/no_think");
                }
                return;
            }
        }
    }
    messages.insert(
        0,
        openai::Message {
            role: "system".to_string(),
            content: Some(openai::MessageContent::Text("/no_think".to_string())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    );
}

fn push_text(blocks: &mut Vec<Value>, text: Option<&str>) {
    if let Some(text) = text {
        let cleaned = sanitize_answer(text);
        if !cleaned.is_empty() {
            blocks.push(json!({"type": "text", "text": cleaned}));
        }
    }
}

/// Clean up an answer from a local model: drop any leaked chain-of-thought (`<think>…</think>`
/// that some backends inline into content) and any leaked Hermes/Qwen tool-call scaffolding
/// (`<tool_call><function=…>` a model emits when it wants a tool but has none), so the client
/// only ever sees the real answer text.
fn sanitize_answer(text: &str) -> String {
    // Keep only what follows the last </think> (the actual answer).
    let text = match text.rfind("</think>") {
        Some(idx) => &text[idx + "</think>".len()..],
        None => text,
    };
    let cut = ["<tool_call>", "<function=", "<|tool_call", "<think>"]
        .iter()
        .filter_map(|m| text.find(m))
        .min()
        .unwrap_or(text.len());
    text[..cut].trim().to_string()
}

fn search_error(srv_id: &str, code: &str) -> Value {
    json!({
        "type": "web_search_tool_result",
        "tool_use_id": srv_id,
        "content": {"type": "web_search_tool_result_error", "error_code": code},
    })
}

fn fetch_error(srv_id: &str, code: &str) -> Value {
    json!({
        "type": "web_fetch_tool_result",
        "tool_use_id": srv_id,
        "content": {"type": "web_fetch_tool_error", "error_code": code},
    })
}

/// A compact, model-readable rendering of search results to feed back as a tool result.
fn format_search(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "No results found.".to_string();
    }
    results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            format!(
                "{}. {} — {}\n{}",
                i + 1,
                r.title.trim(),
                r.url.trim(),
                r.description.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn tool_result_msg(tool_call_id: &str, text: &str) -> openai::Message {
    openai::Message {
        role: "tool".to_string(),
        content: Some(openai::MessageContent::Text(text.to_string())),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: Some(tool_call_id.to_string()),
        name: None,
    }
}

/// Non-streaming backend call across the configured upstreams. Fires immediately and retries
/// (up to [`BACKEND_MAX_ATTEMPTS`]) only on a 5xx or transport error, with a short backoff; a
/// 4xx is deterministic and returned at once.
async fn backend_call(
    client: &Client,
    config: &Config,
    req: &openai::OpenAIRequest,
    api_key: Option<&str>,
) -> ProxyResult<openai::OpenAIResponse> {
    let mut owned = req.clone();
    crate::proxy::normalize_system_first(&mut owned.messages);
    let urls = config.chat_completions_urls();
    let mut last_err = None;
    for url in &urls {
        for attempt in 1..=BACKEND_MAX_ATTEMPTS {
            let mut rb = client.post(url).json(&owned).timeout(Duration::from_secs(600));
            if let Some(key) = api_key {
                rb = rb.header("Authorization", format!("Bearer {key}"));
            }
            match rb.send().await {
                Ok(resp) if resp.status().is_success() => {
                    let bytes = resp.bytes().await.map_err(ProxyError::Http)?;
                    return serde_json::from_slice(&bytes).map_err(|e| {
                        ProxyError::Upstream(format!("invalid upstream response: {e}"))
                    });
                }
                Ok(resp) => {
                    let status = resp.status();
                    let message = resp.text().await.unwrap_or_default();
                    // 4xx is deterministic — surface it immediately, don't burn retries.
                    if !status.is_server_error() {
                        return Err(ProxyError::UpstreamStatus { status, message });
                    }
                    tracing::warn!(%url, %status, attempt, "web agent: backend 5xx, retrying");
                    last_err = Some(ProxyError::UpstreamStatus { status, message });
                }
                Err(err) => {
                    tracing::warn!(%url, attempt, "web agent: backend transport error: {err}");
                    last_err = Some(ProxyError::Http(err));
                }
            }
            // Quick backoff before the next attempt (skip after the last one).
            if attempt < BACKEND_MAX_ATTEMPTS {
                tokio::time::sleep(Duration::from_millis(150 * attempt as u64)).await;
            }
        }
    }
    Err(last_err.unwrap_or_else(|| ProxyError::Upstream("all upstreams failed".into())))
}

fn sse_event(event: &str, data: &Value) -> String {
    format!(
        "event: {event}\ndata: {}\n\n",
        serde_json::to_string(data).unwrap_or_default()
    )
}

/// Replay assembled content blocks as the documented per-block SSE event sequence.
fn render_block_events(blocks: &[Value]) -> Vec<String> {
    let mut frames = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        let ty = block.get("type").and_then(Value::as_str).unwrap_or("");
        match ty {
            "text" => {
                let text = block.get("text").and_then(Value::as_str).unwrap_or("");
                frames.push(sse_event(
                    "content_block_start",
                    &json!({"type": "content_block_start", "index": index,
                            "content_block": {"type": "text", "text": ""}}),
                ));
                frames.push(sse_event(
                    "content_block_delta",
                    &json!({"type": "content_block_delta", "index": index,
                            "delta": {"type": "text_delta", "text": text}}),
                ));
                frames.push(stop_frame(index));
            }
            "server_tool_use" | "tool_use" => {
                let id = block.get("id").and_then(Value::as_str).unwrap_or("");
                let name = block.get("name").and_then(Value::as_str).unwrap_or("");
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                frames.push(sse_event(
                    "content_block_start",
                    &json!({"type": "content_block_start", "index": index,
                            "content_block": {"type": ty, "id": id, "name": name, "input": {}}}),
                ));
                frames.push(sse_event(
                    "content_block_delta",
                    &json!({"type": "content_block_delta", "index": index,
                            "delta": {"type": "input_json_delta",
                                      "partial_json": serde_json::to_string(&input).unwrap_or_default()}}),
                ));
                frames.push(stop_frame(index));
            }
            // Result blocks are emitted whole on the start frame (as Anthropic does).
            _ => {
                frames.push(sse_event(
                    "content_block_start",
                    &json!({"type": "content_block_start", "index": index, "content_block": block}),
                ));
                frames.push(stop_frame(index));
            }
        }
    }
    frames
}

fn stop_frame(index: usize) -> String {
    sse_event(
        "content_block_stop",
        &json!({"type": "content_block_stop", "index": index}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::anthropic;

    fn req_with_tools(tools: Vec<anthropic::Tool>) -> anthropic::AnthropicRequest {
        serde_json::from_value(json!({
            "model": "claude-opus-4-8",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools,
        }))
        .unwrap()
    }

    fn tool(ty: Option<&str>, name: &str) -> anthropic::Tool {
        anthropic::Tool {
            name: name.to_string(),
            description: None,
            input_schema: if ty.is_some() {
                Value::Null
            } else {
                json!({"type": "object"})
            },
            tool_type: ty.map(str::to_string),
        }
    }

    #[test]
    fn detect_finds_search_and_fetch_across_versions() {
        // No web tools → false (normal path).
        assert!(!detect(&req_with_tools(vec![tool(None, "Bash")])));
        // Either server browsing tool, any version, is detected.
        assert!(detect(&req_with_tools(vec![
            tool(None, "Bash"),
            tool(Some("web_search_20250305"), "web_search"),
        ])));
        assert!(detect(&req_with_tools(vec![tool(
            Some("web_fetch_20260209"),
            "web_fetch"
        )])));
    }

    #[test]
    fn rewrite_turns_server_tools_into_callable_functions() {
        let mut req = req_with_tools(vec![
            tool(None, "Bash"),
            tool(Some("web_search_20250305"), "web_search"),
            tool(Some("web_fetch_20250910"), "web_fetch"),
        ]);
        rewrite_tools(&mut req);
        let tools = req.tools.unwrap();
        // Client tool untouched.
        assert_eq!(tools[0].name, "Bash");
        assert!(tools[0].tool_type.is_none());
        // web_search → callable function with a query schema, server type stripped.
        assert_eq!(tools[1].name, "web_search");
        assert!(tools[1].tool_type.is_none());
        assert_eq!(
            tools[1].input_schema["properties"]["query"]["type"],
            "string"
        );
        // web_fetch → callable function with a url schema.
        assert_eq!(tools[2].name, "web_fetch");
        assert_eq!(tools[2].input_schema["properties"]["url"]["type"], "string");
    }

    #[test]
    fn render_emits_full_block_sequence_for_a_search_turn() {
        let blocks = vec![
            json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
                   "input": {"query": "rust"}}),
            json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
                   "content": [{"type": "web_search_result", "url": "https://r.org", "title": "Rust"}]}),
            json!({"type": "text", "text": "Rust is a language."}),
        ];
        let sse = render_block_events(&blocks).join("");
        // server_tool_use streams its input as input_json_delta.
        assert!(sse.contains("\"type\":\"server_tool_use\""));
        assert!(sse.contains("input_json_delta"));
        assert!(sse.contains("\\\"query\\\":\\\"rust\\\""));
        // result block is emitted whole on its start frame.
        assert!(sse.contains("web_search_tool_result"));
        assert!(sse.contains("web_search_result"));
        // final text is delta'd and every block is closed (one stop per block).
        assert!(sse.contains("Rust is a language."));
        assert_eq!(sse.matches("\"type\":\"content_block_stop\"").count(), 3);
    }

    #[test]
    fn search_and_fetch_errors_use_documented_shapes() {
        assert_eq!(
            search_error("srvtoolu_1", "max_uses_exceeded")["content"]["type"],
            "web_search_tool_result_error"
        );
        assert_eq!(
            fetch_error("srvtoolu_1", "url_not_accessible")["content"]["type"],
            "web_fetch_tool_error"
        );
    }
}
