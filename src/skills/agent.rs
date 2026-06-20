//! Proxy-injected tools: `recall_skills` (query the learned-skill store) and `search_docs`
//! (query the self-hosted docs-mcp server). When enabled, the proxy injects these as callable
//! function tools and drives a tool loop itself — exactly like the web-search emulation — so the
//! model can pull learned lessons / up-to-date library docs on demand, transparently to the client
//! (which never sees these tool calls). Client-side tools (Bash, Read, …) are handed back
//! untouched the moment the model calls one.
//!
//! Like the web agent, the loop is non-streaming internally and the result is replayed to the
//! client (as a message, or as SSE with keep-alive heartbeats). Failures degrade gracefully — a
//! tool that errors returns an error string the model can react to; the request never breaks.

use super::{embed, store};
use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::models::openai;
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
use std::time::{Duration, Instant};

const MAX_ROUNDS: usize = 4;
const MAX_TOOL_USES: u32 = 6;
const RECALL_LIMIT: u64 = 5;
const DOCS_LIMIT: u64 = 5;
const DOCS_TIMEOUT: Duration = Duration::from_secs(30);

static ID_COUNTER: AtomicU64 = AtomicU64::new(1);
fn next_id(prefix: &str) -> String {
    format!("{prefix}_{:020x}", ID_COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Whether the proxy-tools loop should run for this config (tools on + at least the skill store).
pub fn enabled(config: &Config) -> bool {
    config.skills.tools && config.skills.enabled
}

/// Inject the proxy-handled function tools into an already-translated request. `recall_skills` is
/// always added (queries the skill store); `search_docs` only when a docs-mcp endpoint is set.
pub fn inject_tools(config: &Config, req: &mut openai::OpenAIRequest) {
    let mut tools = req.tools.take().unwrap_or_default();
    tools.push(function_tool(
        "recall_skills",
        "Recall previously-learned engineering lessons/skills relevant to a query, from the \
         assistant's own accumulated experience. Use when a task resembles past work.",
        json!({
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What you need guidance on."}},
            "required": ["query"]
        }),
    ));
    if config.skills.docs_mcp_url.is_some() {
        tools.push(function_tool(
            "search_docs",
            "Search up-to-date, version-specific documentation for a library/framework. Use to \
             confirm current API usage instead of relying on memory.",
            json!({
                "type": "object",
                "properties": {
                    "library": {"type": "string", "description": "Library name, e.g. 'tokio'."},
                    "query": {"type": "string", "description": "What to look up."}
                },
                "required": ["library", "query"]
            }),
        ));
    }
    req.tools = Some(tools);
}

fn function_tool(name: &str, description: &str, parameters: Value) -> openai::Tool {
    openai::Tool {
        tool_type: "function".to_string(),
        function: openai::Function {
            name: name.to_string(),
            description: Some(description.to_string()),
            parameters,
        },
    }
}

fn is_proxy_tool(name: &str) -> bool {
    matches!(name, "recall_skills" | "search_docs")
}

struct AgentOutput {
    model: String,
    blocks: Vec<Value>,
    stop_reason: String,
    input_tokens: u32,
    output_tokens: u32,
}

/// Entry point: drive the loop, then render as a message (non-streaming) or SSE (streaming).
pub async fn handle(
    config: &Config,
    client: &Client,
    openai_req: openai::OpenAIRequest,
    api_key: Option<&str>,
    streaming: bool,
) -> ProxyResult<Response> {
    if streaming {
        let period = if config.heartbeat_secs == 0 {
            Duration::from_secs(60 * 60 * 24 * 365)
        } else {
            Duration::from_secs(config.heartbeat_secs)
        };
        // Own the data the stream needs.
        let config = config.clone();
        let client = client.clone();
        let api_key = api_key.map(str::to_string);
        let stream = sse_stream(config, client, openai_req, api_key, period);
        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", HeaderValue::from_static("text/event-stream"));
        headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
        headers.insert("Connection", HeaderValue::from_static("keep-alive"));
        Ok((headers, Body::from_stream(stream)).into_response())
    } else {
        let out = run_loop(config, client, openai_req, api_key).await?;
        Ok(Json(json!({
            "id": next_id("msg"),
            "type": "message",
            "role": "assistant",
            "model": out.model,
            "content": out.blocks,
            "stop_reason": out.stop_reason,
            "stop_sequence": null,
            "usage": {"input_tokens": out.input_tokens, "output_tokens": out.output_tokens},
        }))
        .into_response())
    }
}

/// SSE: announce the message, beat keep-alives while the loop runs, then replay the blocks.
fn sse_stream(
    config: Config,
    client: Client,
    openai_req: openai::OpenAIRequest,
    api_key: Option<String>,
    period: Duration,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let msg_id = next_id("msg");
        let model = openai_req.model.clone();
        yield Ok(Bytes::from(sse_event("message_start", &json!({
            "type": "message_start",
            "message": {"id": msg_id, "type": "message", "role": "assistant", "model": model,
                        "content": [], "stop_reason": null, "stop_sequence": null,
                        "usage": {"input_tokens": 0, "output_tokens": 0}}
        }))));

        let fut = run_loop(&config, &client, openai_req, api_key.as_deref());
        tokio::pin!(fut);
        let mut beat = tokio::time::interval(period);
        beat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        beat.tick().await;

        let out = loop {
            tokio::select! {
                r = &mut fut => break r,
                _ = beat.tick() => { yield Ok(Bytes::from_static(b": keep-alive\n\n")); }
            }
        };

        match out {
            Ok(out) => {
                for frame in render_block_events(&out.blocks) { yield Ok(Bytes::from(frame)); }
                yield Ok(Bytes::from(sse_event("message_delta", &json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": out.stop_reason, "stop_sequence": null},
                    "usage": {"input_tokens": out.input_tokens, "output_tokens": out.output_tokens}
                }))));
                yield Ok(Bytes::from(sse_event("message_stop", &json!({"type": "message_stop"}))));
            }
            Err(err) => {
                yield Ok(Bytes::from(sse_event("error", &json!({
                    "type": "error", "error": {"type": "api_error", "message": err.to_string()}
                }))));
            }
        }
    }
}

/// Drive model↔tool rounds: execute `recall_skills`/`search_docs` ourselves; stop the moment the
/// model answers or calls a client-side tool (which we hand back).
async fn run_loop(
    config: &Config,
    client: &Client,
    mut req: openai::OpenAIRequest,
    api_key: Option<&str>,
) -> ProxyResult<AgentOutput> {
    // Keep the client's max_tokens — every request flows through here, so capping would truncate
    // long outputs. Only the internal rounds are non-streaming.
    req.stream = Some(false);
    req.stream_options = None;
    let mut messages = req.messages.clone();
    let mut blocks: Vec<Value> = Vec::new();
    let mut out = AgentOutput {
        model: req.model.clone(),
        blocks: Vec::new(),
        stop_reason: "end_turn".to_string(),
        input_tokens: 0,
        output_tokens: 0,
    };
    let mut uses: u32 = 0;

    for round in 0..MAX_ROUNDS {
        let mut r = req.clone();
        r.messages = messages.clone();
        // Final round (or budget spent): drop our tools so the model must answer.
        let force_answer = round + 1 >= MAX_ROUNDS || uses >= MAX_TOOL_USES;
        if force_answer {
            strip_proxy_tools(&mut r);
        }

        let resp = backend_call(client, config, &r, api_key).await?;
        if let Some(m) = &resp.model {
            out.model = m.clone();
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
        let all_proxy = !tool_calls.is_empty()
            && tool_calls.iter().all(|tc| is_proxy_tool(&tc.function.name));

        // Terminal: a plain answer, or a client-side tool call we must hand back.
        if !all_proxy {
            // Preserve extended-thinking output (precedes the visible answer).
            if let Some(t) = &msg.reasoning_content {
                if !t.is_empty() {
                    blocks.push(json!({"type": "thinking", "thinking": t}));
                }
            }
            push_text(&mut blocks, msg.content.as_deref());
            if tool_calls.is_empty() {
                out.stop_reason = "end_turn".to_string();
            } else {
                for tc in &tool_calls {
                    blocks.push(json!({
                        "type": "tool_use", "id": tc.id, "name": tc.function.name,
                        "input": parse_args(&tc.function.arguments)
                    }));
                }
                out.stop_reason = "tool_use".to_string();
            }
            out.blocks = blocks;
            return Ok(out);
        }

        // The model wants our tools: keep its turn, run each call, feed results back.
        messages.push(openai::Message {
            role: "assistant".to_string(),
            content: msg.content.clone().map(openai::MessageContent::Text),
            reasoning_content: None,
            tool_calls: Some(tool_calls.clone()),
            tool_call_id: None,
            name: None,
        });
        for tc in &tool_calls {
            uses += 1;
            let args = parse_args(&tc.function.arguments);
            let t = Instant::now();
            let result = execute(config, client, &tc.function.name, &args, api_key).await;
            tracing::info!(
                tool = %tc.function.name, ms = t.elapsed().as_millis() as u64,
                "skills/agent: tool executed"
            );
            messages.push(tool_result_msg(&tc.id, &result));
        }
    }

    out.blocks = blocks;
    Ok(out)
}

/// Execute one proxy tool, returning a model-readable result string (best-effort).
async fn execute(
    config: &Config,
    client: &Client,
    name: &str,
    args: &Value,
    api_key: Option<&str>,
) -> String {
    match name {
        "recall_skills" => {
            let query = args.get("query").and_then(Value::as_str).unwrap_or("");
            recall_skills(config, client, query, api_key).await
        }
        "search_docs" => {
            let library = args.get("library").and_then(Value::as_str).unwrap_or("");
            let query = args.get("query").and_then(Value::as_str).unwrap_or("");
            search_docs(config, client, library, query).await
        }
        other => format!("Unknown tool: {other}"),
    }
}

/// recall_skills: embed the query and return the most relevant injectable skills.
async fn recall_skills(config: &Config, client: &Client, query: &str, api_key: Option<&str>) -> String {
    if query.trim().is_empty() {
        return "No query provided.".to_string();
    }
    let Some(vector) = embed::embed(config, client, query, api_key).await else {
        return "Skill recall unavailable (embedding failed).".to_string();
    };
    let qc = store::QdrantClient::new(
        config.skills.qdrant_url.clone(),
        config.skills.collection.clone(),
        client.clone(),
    );
    let found = qc
        .search(
            &vector,
            RECALL_LIMIT as u32,
            config.skills.min_score,
            &config.skills.inject_tiers,
        )
        .await
        .unwrap_or_default();
    if found.is_empty() {
        return "No relevant learned skills found.".to_string();
    }
    found
        .iter()
        .enumerate()
        .map(|(i, s)| format!("{}. {} — {}", i + 1, s.payload.title, s.payload.body))
        .collect::<Vec<_>>()
        .join("\n")
}

/// search_docs: query the docs-mcp server (MCP Streamable HTTP, stateless) for library docs.
async fn search_docs(config: &Config, client: &Client, library: &str, query: &str) -> String {
    let Some(url) = &config.skills.docs_mcp_url else {
        return "Docs search not configured.".to_string();
    };
    if library.trim().is_empty() || query.trim().is_empty() {
        return "Provide both 'library' and 'query'.".to_string();
    }
    let body = json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search_docs", "arguments": {
            "library": library, "query": query, "limit": DOCS_LIMIT
        }}
    });
    let resp = match client
        .post(url)
        .timeout(DOCS_TIMEOUT)
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream")
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return format!("Docs search failed: {e}"),
    };
    let text = resp.text().await.unwrap_or_default();
    // Response is an SSE-framed JSON-RPC result; take the first data line.
    let data = text
        .lines()
        .find_map(|l| l.strip_prefix("data: "))
        .unwrap_or_else(|| text.trim());
    let parsed: Value = serde_json::from_str(data).unwrap_or(Value::Null);
    parsed
        .pointer("/result/content/0/text")
        .and_then(Value::as_str)
        .map(|s| s.chars().take(6000).collect::<String>())
        .unwrap_or_else(|| "No documentation found.".to_string())
}

fn strip_proxy_tools(req: &mut openai::OpenAIRequest) {
    if let Some(tools) = req.tools.as_mut() {
        tools.retain(|t| !is_proxy_tool(&t.function.name));
        if tools.is_empty() {
            req.tools = None;
            req.tool_choice = None;
        }
    }
}

fn parse_args(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({}))
}

fn push_text(blocks: &mut Vec<Value>, text: Option<&str>) {
    if let Some(t) = text {
        if !t.trim().is_empty() {
            blocks.push(json!({"type": "text", "text": t}));
        }
    }
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

/// Non-streaming backend call (immediate, retry only on 5xx/transport).
async fn backend_call(
    client: &Client,
    config: &Config,
    req: &openai::OpenAIRequest,
    api_key: Option<&str>,
) -> ProxyResult<openai::OpenAIResponse> {
    let urls = config.chat_completions_urls();
    let mut last = None;
    for url in &urls {
        for attempt in 1..=3u32 {
            let mut rb = client.post(url).json(req).timeout(Duration::from_secs(600));
            if let Some(key) = api_key {
                rb = rb.header("Authorization", format!("Bearer {key}"));
            }
            match rb.send().await {
                Ok(resp) if resp.status().is_success() => {
                    let bytes = resp.bytes().await.map_err(ProxyError::Http)?;
                    return serde_json::from_slice(&bytes)
                        .map_err(|e| ProxyError::Upstream(format!("invalid upstream response: {e}")));
                }
                Ok(resp) => {
                    let status = resp.status();
                    let message = resp.text().await.unwrap_or_default();
                    if !status.is_server_error() {
                        return Err(ProxyError::UpstreamStatus { status, message });
                    }
                    last = Some(ProxyError::UpstreamStatus { status, message });
                }
                Err(err) => last = Some(ProxyError::Http(err)),
            }
            if attempt < 3 {
                tokio::time::sleep(Duration::from_millis(150 * attempt as u64)).await;
            }
        }
    }
    Err(last.unwrap_or_else(|| ProxyError::Upstream("all upstreams failed".into())))
}

fn sse_event(event: &str, data: &Value) -> String {
    format!("event: {event}\ndata: {}\n\n", serde_json::to_string(data).unwrap_or_default())
}

/// Replay assembled content blocks (text / tool_use) as the documented SSE event sequence.
fn render_block_events(blocks: &[Value]) -> Vec<String> {
    let mut frames = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        match block.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => {
                let text = block.get("text").and_then(Value::as_str).unwrap_or("");
                frames.push(sse_event("content_block_start", &json!({
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "text", "text": ""}})));
                frames.push(sse_event("content_block_delta", &json!({
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "text_delta", "text": text}})));
                frames.push(stop_frame(index));
            }
            "tool_use" => {
                let id = block.get("id").and_then(Value::as_str).unwrap_or("");
                let name = block.get("name").and_then(Value::as_str).unwrap_or("");
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                frames.push(sse_event("content_block_start", &json!({
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}})));
                frames.push(sse_event("content_block_delta", &json!({
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta",
                              "partial_json": serde_json::to_string(&input).unwrap_or_default()}})));
                frames.push(stop_frame(index));
            }
            "thinking" => {
                let thinking = block.get("thinking").and_then(Value::as_str).unwrap_or("");
                frames.push(sse_event("content_block_start", &json!({
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "thinking", "thinking": ""}})));
                frames.push(sse_event("content_block_delta", &json!({
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "thinking_delta", "thinking": thinking}})));
                frames.push(stop_frame(index));
            }
            _ => {}
        }
    }
    frames
}

fn stop_frame(index: usize) -> String {
    sse_event("content_block_stop", &json!({"type": "content_block_stop", "index": index}))
}
