//! Stage 2: distil reusable lessons from conversation history into `candidate` skills.
//!
//! Clients (e.g. Claude Code) resend the full conversation each turn, so a request's message
//! history *is* the transcript. After responding we distil from it in a background task, throttled
//! per-conversation so a growing thread isn't re-distilled every turn. An LLM judge (temp 0)
//! labels the outcome and extracts ≤3 GENERAL lessons — learning from both success (validated
//! strategies) and failure ("avoid this") — following FAIL-unless-proven: it must not treat the
//! mere absence of a complaint as success. Each lesson is embedded and written as `candidate`,
//! which is NOT injectable until Stage 3 verification promotes it.

use crate::config::Config;
use crate::models::anthropic;
use super::{embed, llm, store};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

const MIN_MESSAGES: usize = 4; // need a real exchange, not a one-shot
const DISTILL_STRIDE: usize = 3; // re-distil a growing thread every N new messages (LLM is unmetered)
const MAX_TRANSCRIPT_CHARS: usize = 12_000;
const MAX_LESSONS: usize = 3;
// Room for the model's (often unsuppressed) reasoning_content PLUS up to 3 JSON lessons in
// content — a tight budget truncates mid-thought and yields empty/invalid output.
const JUDGE_MAX_TOKENS: u32 = 3072;
/// A new lesson this similar to a promoted skill is a contradiction candidate worth an LLM check.
const CONTRA_MIN_SCORE: f32 = 0.82;
/// Write-side near-identical dedup (Qwen3-Embedding-4B): a new lesson this similar to an existing
/// skill reuses/skips it instead of stacking a paraphrase copy. Conservative (cosine-only, no LLM).
const SKILL_DEDUP_THRESHOLD: f32 = 0.90;
const CONTRA_MAX_TOKENS: u32 = 2048;
const CONTRA_SYSTEM: &str = "You compare two engineering lessons for an AI coding assistant. Decide \
whether lesson B CONTRADICTS lesson A — i.e. gives opposite or incompatible advice for the same \
situation (not merely overlapping, broader, or unrelated). Output STRICT JSON only, no prose: \
{\"contradict\": true|false}";

/// Per-conversation throttle: signature -> message count at last distillation.
fn tracker() -> &'static Mutex<HashMap<u64, usize>> {
    static T: OnceLock<Mutex<HashMap<u64, usize>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Decide whether this request's history is worth distilling now; if so, spawn the work.
/// Cheap and synchronous — returns immediately, all real work is off the request path.
pub fn maybe_spawn(
    config: Arc<Config>,
    client: Client,
    req: &anthropic::AnthropicRequest,
    api_key: Option<String>,
) {
    if !config.skills.learn {
        return;
    }
    let n = req.messages.len();
    if n < MIN_MESSAGES {
        return;
    }
    let sig = conversation_signature(req);
    {
        let mut t = tracker().lock().unwrap();
        let last = t.get(&sig).copied().unwrap_or(0);
        if n < last + DISTILL_STRIDE {
            return;
        }
        t.insert(sig, n);
        if t.len() > 10_000 {
            t.clear(); // crude bound; throttling is best-effort
        }
    }
    let transcript = render_transcript(req);
    if transcript.trim().is_empty() {
        return;
    }
    tracing::debug!(messages = n, "skills/distill: spawning");
    tokio::spawn(async move {
        distill(&config, &client, &transcript, api_key.as_deref()).await;
    });
}

#[derive(Deserialize)]
struct Judgement {
    #[serde(default)]
    outcome: String,
    #[serde(default)]
    lessons: Vec<Lesson>,
    /// One-line topic summary of the conversation (observability only).
    #[serde(default)]
    summary: String,
    /// Specific, researchable topics/technologies the user engaged with — fed into the proactive
    /// queue so the background loop learns "what users are interested in" (richer than the raw last
    /// question). Empty when nothing is worth researching.
    #[serde(default)]
    interests: Vec<String>,
}

#[derive(Deserialize)]
struct Lesson {
    #[serde(default)]
    title: String,
    #[serde(default)]
    when_to_use: String,
    #[serde(default)]
    body: String,
    #[serde(default)]
    kind: String,
}

const JUDGE_SYSTEM: &str = "You are a meticulous engineering-knowledge distiller for an AI coding assistant. \
The conversation transcript includes tool calls and their results. Do two things. (1) Judge whether the \
assistant ultimately SUCCEEDED, FAILED, or it is UNCLEAR at the user's request — be strict and default to \
\"unclear\" unless there is clear evidence; the mere absence of a user complaint is NOT success. \
(2) Extract at most 3 concise, GENERAL, reusable lessons that would help on FUTURE similar tasks. Learn \
from success (validated strategies, kind=positive) AND from failure (preventive 'avoid this' lessons, \
kind=negative). PAY SPECIAL ATTENTION to cases where a command or tool call FAILED (an error in its result) \
and a later attempt SUCCEEDED: distil the corrected usage into a 'how to use it correctly next time' lesson \
— state the right form and the pitfall it avoids (kind=positive). Keep every lesson GENERAL and \
transferable: never include secrets, credentials, file contents, names, paths, or other task-specific \
details. Be conservative: if the conversation is trivial, off-topic, or you cannot tell, return an empty \
lessons array. (3) Write a one-line 'summary' of the topic, and list up to 3 'interests' — SPECIFIC, \
researchable topics or technologies the user engaged with (e.g. 'latest Bun version', 'Rust async \
cancellation'), suitable for later background web research; leave 'interests' empty if nothing is worth \
researching (small talk, purely local edits). Output STRICT JSON only, no prose: \
{\"outcome\":\"success|failure|unclear\",\"summary\":\"one line\",\"interests\":[\"researchable topic\"],\"lessons\":[{\"title\":\"\",\"when_to_use\":\"short trigger phrase\",\"body\":\"actionable lesson\",\"kind\":\"positive|negative\"}]}";

async fn distill(config: &Config, client: &Client, transcript: &str, api_key: Option<&str>) {
    let user = format!("Transcript:\n{transcript}\n\nReturn the JSON now.");
    tracing::debug!(transcript_len = transcript.len(), "skills/distill: judging");
    let Some(value) =
        llm::chat_json_hard(config, client, JUDGE_SYSTEM, &user, JUDGE_MAX_TOKENS).await
    else {
        tracing::warn!("skills/distill: judge returned no usable JSON");
        return;
    };
    let judged: Judgement = match serde_json::from_value(value) {
        Ok(j) => j,
        Err(e) => {
            tracing::warn!("skills/distill: judgement parse failed: {e}");
            return;
        }
    };

    // Feed researchable interests from this conversation into the proactive queue — a richer signal
    // than the raw last question, so the background loop learns "what users are interested in". Done
    // before the no-lessons early-return, so even a chat with no distillable lesson surfaces topics.
    let interests: Vec<String> = judged
        .interests
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .take(3)
        .collect();
    if config.skills.proactive {
        for topic in &interests {
            super::proactive::record_question(topic);
        }
    }
    if !judged.summary.trim().is_empty() || !interests.is_empty() {
        super::eventlog::record(
            "summary",
            json!({
                "summary": judged.summary.trim(),
                "interests": interests,
                "outcome": judged.outcome,
            }),
        );
    }

    let lessons: Vec<Lesson> = judged
        .lessons
        .into_iter()
        .filter(|l| !l.title.trim().is_empty() && !l.body.trim().is_empty())
        .take(MAX_LESSONS)
        .collect();
    if lessons.is_empty() {
        tracing::debug!(outcome = %judged.outcome, "skills/distill: no lessons");
        return;
    }

    let qc = store::QdrantClient::new(
        config.skills.qdrant_url.clone(),
        config.skills.collection.clone(),
        client.clone(),
    );
    let now = unix_now();
    let mut written = 0;
    for l in &lessons {
        let route = if l.when_to_use.trim().is_empty() {
            l.title.clone()
        } else {
            format!("{} {}", l.when_to_use, l.title)
        };
        let Some(vector) = embed::embed(config, client, &route, api_key).await else {
            continue;
        };
        if !qc.ensure_collection(vector.len()).await {
            continue;
        }
        let kind = if l.kind.trim().is_empty() {
            "positive".to_string()
        } else {
            l.kind.trim().to_ascii_lowercase()
        };
        // Semantic dedup at the source: if a near-identical skill already exists (any tier, even a
        // differently-worded paraphrase), don't stack another copy. If it's already promoted, this
        // candidate adds nothing — skip it (and never demote the promoted one). If it's another
        // candidate, update it in place. Without this, re-distilling the same lesson under slightly
        // different titles piled up dozens of duplicates (`stable_id(title)` keys on the title only).
        let near = qc.search_raw(&vector, 1, SKILL_DEDUP_THRESHOLD, &[], false).await;
        let id = match near.first() {
            Some(h) => {
                let ex_tier = serde_json::from_value::<store::SkillPayload>(h.payload.clone())
                    .map(|p| p.tier)
                    .unwrap_or_default();
                if ex_tier == "verified" || ex_tier == "trusted" {
                    continue; // already covered by a promoted skill
                }
                h.id // reinforce the existing candidate instead of duplicating
            }
            None => store::stable_id(&l.title.to_lowercase()),
        };
        let payload = json!({
            "tier": "candidate",
            "title": l.title,
            "when_to_use": l.when_to_use,
            "body": l.body,
            "kind": kind,
            "source": "distill",
            "outcome": judged.outcome.clone(),
            "created_at": now,
            "updated_at": now,
            "use_count": 0,
            "success_count": 0,
        });
        if qc.upsert(id, &vector, payload).await {
            written += 1;
        }
        // Contradiction check: does this lesson conflict with an existing PROMOTED skill (opposite
        // advice for the same situation)? Surface it rather than silently accumulating both — the
        // re-verify loop then resolves which one actually holds up against the web.
        let near = qc
            .search_raw(
                &vector,
                1,
                CONTRA_MIN_SCORE,
                &["verified".to_string(), "trusted".to_string()],
                false,
            )
            .await;
        if let Some(h) = near.first() {
            if let Ok(ex) = serde_json::from_value::<store::SkillPayload>(h.payload.clone()) {
                if ex.title.to_lowercase() != l.title.to_lowercase()
                    && contradicts(config, client, l, &ex).await
                {
                    tracing::info!(new = %l.title, existing = %ex.title, "skills/distill: contradiction with existing skill");
                    super::eventlog::record(
                        "contradiction",
                        json!({"new": l.title.clone(), "existing": ex.title.clone()}),
                    );
                }
            }
        }
    }
    if written > 0 {
        tracing::info!(written, outcome = %judged.outcome, "skills/distill: wrote candidate lessons");
        super::eventlog::record(
            "distill",
            json!({
                "outcome": judged.outcome,
                "written": written,
                "skills": lessons.iter().map(|l| l.title.clone()).collect::<Vec<_>>(),
            }),
        );
    }
}

#[derive(Deserialize)]
struct ContraVerdict {
    #[serde(default)]
    contradict: bool,
}

/// Ask the (unmetered) LLM whether the new lesson contradicts an existing promoted skill.
async fn contradicts(
    config: &Config,
    client: &Client,
    new_l: &Lesson,
    existing: &store::SkillPayload,
) -> bool {
    let user = format!(
        "Lesson A (existing):\nTitle: {}\nBody: {}\n\nLesson B (new):\nTitle: {}\nBody: {}\n\nDoes B contradict A? Return the JSON.",
        existing.title, existing.body, new_l.title, new_l.body
    );
    matches!(
        llm::chat_json(
            config,
            client,
            CONTRA_SYSTEM,
            &user,
            super::background_api_key(config).as_deref(),
            CONTRA_MAX_TOKENS,
        )
        .await
        .and_then(|v| serde_json::from_value::<ContraVerdict>(v).ok()),
        Some(ContraVerdict { contradict: true })
    )
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A conversation's identity across its turns = a hash of its first user message.
fn conversation_signature(req: &anthropic::AnthropicRequest) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for m in &req.messages {
        if m.role == "user" {
            text_of(&m.content).hash(&mut h);
            break;
        }
    }
    h.finish()
}

fn render_transcript(req: &anthropic::AnthropicRequest) -> String {
    let mut out = String::new();
    for m in &req.messages {
        let body = text_of(&m.content);
        if body.trim().is_empty() {
            continue;
        }
        out.push_str(&m.role);
        out.push_str(": ");
        out.push_str(body.trim());
        out.push_str("\n\n");
        if out.len() > MAX_TRANSCRIPT_CHARS {
            break;
        }
    }
    if out.len() > MAX_TRANSCRIPT_CHARS {
        out.truncate(MAX_TRANSCRIPT_CHARS);
    }
    out
}

/// Flatten a message's content to text (tool uses/results summarized; images skipped).
fn text_of(content: &anthropic::MessageContent) -> String {
    match content {
        anthropic::MessageContent::Text(t) => t.clone(),
        anthropic::MessageContent::Blocks(blocks) => {
            let mut parts = Vec::new();
            for b in blocks {
                match b {
                    anthropic::ContentBlock::Text { text, .. } => parts.push(text.clone()),
                    anthropic::ContentBlock::Thinking { thinking } => parts.push(thinking.clone()),
                    anthropic::ContentBlock::ToolUse { name, input, .. } => {
                        parts.push(format!("[tool_use {name}: {}]", truncate(&input.to_string(), 500)))
                    }
                    anthropic::ContentBlock::ToolResult { content, .. } => {
                        parts.push(format!("[tool_result: {}]", truncate(&tool_result_text(content), 800)))
                    }
                    anthropic::ContentBlock::Image { .. } => {}
                }
            }
            parts.join("\n")
        }
    }
}

fn tool_result_text(content: &anthropic::ToolResultContent) -> String {
    match content {
        anthropic::ToolResultContent::Text(t) => t.clone(),
        anthropic::ToolResultContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                anthropic::ContentBlock::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect()
    }
}
