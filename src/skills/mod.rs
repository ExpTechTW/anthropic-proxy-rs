//! Self-learning skill injection.
//!
//! Stage 1 (this file's public surface): on each incoming request, embed the user's ask, retrieve
//! the most relevant **trusted/verified** skills from Qdrant, and inject them as an extra system
//! block — transparently to the client, and best-effort so a failure never blocks the request.
//!
//! Later stages add the write side (distillation from finished conversations), a verification /
//! trust-tier gate, curation, and proactive self-study — all writing into the same Qdrant store
//! that this read path queries. Retrieval here is already restricted to the injectable tiers
//! (`config.skills.inject_tiers`), so candidate/unverified knowledge is never surfaced.
//!
//! Design constraints honoured: no new crates (Qdrant + embeddings over REST), top-k capped (a
//! large store must not flood the prompt — over-injection degrades quality), and graceful
//! degradation everywhere.

mod agent;
mod curate;
mod distill;
mod docs;
mod embed;
mod facts;
mod eventlog;
mod llm;
mod proactive;
mod reflect;
mod store;
mod verify;

pub use agent::{
    enabled as agent_enabled, handle as agent_handle, inject_tools as agent_inject_tools,
};
pub use docs::relevant_docs;

#[allow(unused_imports)]
pub use store::{stable_id, QdrantClient};
pub use distill::maybe_spawn as maybe_spawn_distill;
pub use verify::spawn as spawn_verify;
pub use curate::spawn as spawn_curate;
pub use proactive::{record_question, spawn as spawn_proactive};
pub use facts::spawn_validity as spawn_facts_validity;
pub use facts::relevant_facts;
pub use reflect::spawn as spawn_reflect;
pub use eventlog::record as log_event;

/// Start the compact learning-event log (no-op unless `ANTHROPIC_PROXY_SKILLS_EVENTLOG_PATH` set).
pub fn init_eventlog(config: &Config) {
    eventlog::init(
        &config.skills.eventlog_path,
        config.skills.eventlog_retention_days,
    );
}

use crate::config::Config;
use crate::models::anthropic;
use reqwest::Client;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

// SearXNG is the one rate-limited resource (engines CAPTCHA under load), so ALL searches —
// interactive `web_search` AND background learning — share ONE global budget, paced evenly to a
// sustainable total rate. The budget is split fairly 50/50 between the two lanes WHEN BOTH are busy,
// but is **work-conserving**: if one lane is idle the other uses the whole budget (capacity is never
// wasted). Token-by-token even pacing, never a burst-then-idle.
/// Min gap between any two searches (total rate cap across both lanes).
const SEARCH_TOTAL_GAP: Duration = Duration::from_millis(2000);
/// A lane held to its 50% share gets every other slot (2× the total gap).
const SEARCH_LANE_GAP: Duration = Duration::from_millis(4000);
/// The other lane counts as "active" (→ enforce the 50/50 split) if its last grant is this recent
/// (or still queued ahead).
const SEARCH_FAIR_WINDOW: Duration = Duration::from_millis(4000);
/// Idle guard: background search won't run until this long after the most recent USER search, so
/// the (now reliable but still IP-limited) search budget goes to real queries first and learning
/// fills the idle gaps. Lets the learning loops be aggressive without ever delaying a user query.
const SEARCH_BG_YIELD: Duration = Duration::from_secs(12);

/// Which half of the SearXNG budget a search draws from.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum SearchLane {
    /// Interactive `web_search` emulation (client-facing).
    User,
    /// Background learning loops (verify / proactive / facts / reflect).
    Background,
}

struct GateState {
    global_last: Option<Instant>,
    user_last: Option<Instant>,
    bg_last: Option<Instant>,
}

/// Reserve this lane's next search slot and wait for it. Enforces the global even pace; while the
/// OTHER lane is also active, holds this lane to its 50% share — otherwise lets it use the full
/// budget (work-conserving). Lock is held only for the arithmetic, never across the await.
pub(crate) async fn search_gate(lane: SearchLane) {
    static STATE: OnceLock<Mutex<GateState>> = OnceLock::new();
    let now = Instant::now();
    let grant = {
        let mut s = STATE
            .get_or_init(|| {
                Mutex::new(GateState {
                    global_last: None,
                    user_last: None,
                    bg_last: None,
                })
            })
            .lock()
            .unwrap();
        // Global even pace (total-rate cap across both lanes).
        let mut grant = match s.global_last {
            Some(t) => (t + SEARCH_TOTAL_GAP).max(now),
            None => now,
        };
        let (my_last, other_last) = match lane {
            SearchLane::User => (s.user_last, s.bg_last),
            SearchLane::Background => (s.bg_last, s.user_last),
        };
        // Only throttle this lane to its 50% share while the other lane is active (recently granted
        // or queued ahead); if the other lane is idle, skip it so this lane uses the whole budget.
        let other_active = other_last.is_some_and(|t| t + SEARCH_FAIR_WINDOW >= now);
        if other_active {
            if let Some(t) = my_last {
                grant = grant.max(t + SEARCH_LANE_GAP);
            }
        }
        // Background yields hard to users: hold off until SEARCH_BG_YIELD past the last USER search,
        // so learning runs in the idle gaps and never competes with a real query for the egress IPs.
        if matches!(lane, SearchLane::Background) {
            if let Some(t) = s.user_last {
                grant = grant.max(t + SEARCH_BG_YIELD);
            }
        }
        s.global_last = Some(grant);
        match lane {
            SearchLane::User => s.user_last = Some(grant),
            SearchLane::Background => s.bg_last = Some(grant),
        }
        grant
    };
    let delay = grant.checked_duration_since(now).unwrap_or(Duration::ZERO);
    if !delay.is_zero() {
        tokio::time::sleep(delay).await;
    }
}

/// Remember the most recent non-empty client API key so background learning loops (which run off
/// the request path and have no client key) can still call the upstream in passthrough mode.
fn remembered_key() -> &'static Mutex<Option<String>> {
    static K: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    K.get_or_init(|| Mutex::new(None))
}

pub fn remember_api_key(key: Option<&str>) {
    if let Some(k) = key {
        if !k.is_empty() {
            *remembered_key().lock().unwrap() = Some(k.to_string());
        }
    }
}

/// The API key background tasks should use: the configured one, else the last-seen client key.
#[allow(dead_code)]
pub fn background_api_key(config: &Config) -> Option<String> {
    config
        .skills
        .api_key
        .clone()
        .or_else(|| remembered_key().lock().unwrap().clone())
}

/// Web search for the learning loops (verify corroboration, proactive research). Prefers a
/// configured SearXNG instance (reliable, 70+ engines, deduped) and falls back to open-websearch.
/// The corroboration source must be reliable: when it errors, `verify` returns a transient `None`
/// and the candidate is retried forever without a verdict — flaky DuckDuckGo scraping otherwise
/// strands candidates in the `candidate` tier (never promoted, never injectable).
pub(crate) async fn web_search(
    config: &Config,
    client: &Client,
    query: &str,
    limit: u32,
) -> anyhow::Result<Vec<crate::websearch::SearchResult>> {
    search_gate(SearchLane::Background).await; // background half of the shared SearXNG budget
    if let Some(url) = &config.searxng_url {
        match crate::searx::SearxClient::new(url.clone(), client.clone())
            .search(query, limit)
            .await
        {
            Ok(r) => return Ok(r),
            Err(e) => {
                tracing::debug!("skills/web_search: searxng failed ({e}); falling back to open-websearch")
            }
        }
    }
    crate::websearch::WebSearchClient::new(config.websearch_url.clone(), client.clone())
        .search(query, limit, &["duckduckgo".to_string()], "request")
        .await
}

/// A skill chosen for injection, carrying what we need to render it and report it back.
#[derive(Debug, Clone)]
pub struct RetrievedSkill {
    pub id: String,
    pub title: String,
    pub body: String,
    #[allow(dead_code)]
    pub score: f32,
}

/// Candidate pool for MMR — over-fetch, then diversify down to top_k.
const MMR_POOL: u32 = 12;
/// MMR relevance/diversity tradeoff (higher favours relevance).
const MMR_LAMBDA: f32 = 0.7;

/// Maximal Marginal Relevance (Carbonell & Goldstein 1998): greedily pick `k` items that are
/// relevant to the query (high score) yet dissimilar to those already picked — so we never spend
/// scarce injection slots on near-duplicate lessons (redundant context dilutes attention, and
/// retrieval quality is the dominant factor in memory utility). Naturally injects fewer when fewer
/// candidates clear `min_score`.
fn mmr_select(mut pool: Vec<store::RawScored>, k: usize) -> Vec<store::RawScored> {
    let mut selected: Vec<store::RawScored> = Vec::new();
    while selected.len() < k && !pool.is_empty() {
        let mut best = 0usize;
        let mut best_mmr = f32::MIN;
        for (i, c) in pool.iter().enumerate() {
            let max_sim = selected
                .iter()
                .filter_map(|s| match (&c.vector, &s.vector) {
                    (Some(a), Some(b)) => Some(cosine(a, b)),
                    _ => None,
                })
                .fold(0.0f32, f32::max);
            let mmr = MMR_LAMBDA * c.score - (1.0 - MMR_LAMBDA) * max_sim;
            if mmr > best_mmr {
                best_mmr = mmr;
                best = i;
            }
        }
        selected.push(pool.remove(best));
    }
    selected
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for k in 0..a.len() {
        dot += a[k] * b[k];
        na += a[k] * a[k];
        nb += b[k] * b[k];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Retrieve the top relevant injectable skills for `query`. Returns an empty vec when the feature
/// is disabled, the query is empty, or anything fails — the request then proceeds untouched.
pub async fn retrieve(
    config: &Config,
    client: &Client,
    query: &str,
    api_key: Option<&str>,
) -> Vec<RetrievedSkill> {
    if !config.skills.enabled || query.trim().is_empty() {
        return Vec::new();
    }
    let Some(vector) = embed::embed(config, client, query, api_key).await else {
        return Vec::new();
    };
    let qc = store::QdrantClient::new(
        config.skills.qdrant_url.clone(),
        config.skills.collection.clone(),
        client.clone(),
    );
    // Over-fetch a candidate pool WITH vectors, then MMR-select top_k (relevant + diverse).
    let pool = qc
        .search_raw(
            &vector,
            MMR_POOL,
            config.skills.min_score,
            &config.skills.inject_tiers,
            true,
        )
        .await;
    mmr_select(pool, config.skills.top_k as usize)
        .into_iter()
        .filter_map(|s| {
            let p: store::SkillPayload = serde_json::from_value(s.payload).ok()?;
            Some(RetrievedSkill {
                id: s.id.to_string(),
                title: p.title,
                body: p.body,
                score: s.score,
            })
        })
        .collect()
}

/// Append the retrieved skills to the request's system prompt as one extra `text` block, placed
/// **after** the client's own system content so the (large, stable) client prefix is undisturbed
/// — important for the upstream's prefix cache. Returns the injected skill ids (for the
/// transparency header / log). No-op when `skills` is empty.
pub fn inject(req: &mut anthropic::AnthropicRequest, skills: &[RetrievedSkill]) -> Vec<String> {
    if skills.is_empty() {
        return Vec::new();
    }

    let mut rendered = String::from(
        "# Auto-injected learned knowledge\n\
         The following lessons were retrieved as relevant to the current task. They are learned \
         from past experience and may be imperfect — apply them when they help and ignore anything \
         that does not fit.\n",
    );
    for s in skills {
        rendered.push_str("\n## ");
        rendered.push_str(&s.title);
        rendered.push('\n');
        rendered.push_str(&s.body);
        rendered.push('\n');
    }

    append_system_block(req, rendered);
    skills.iter().map(|s| s.id.clone()).collect()
}

/// Inject retrieved library documentation as a system block (streaming-preserving push). No-op
/// when empty.
pub fn inject_docs(req: &mut anthropic::AnthropicRequest, docs: &str) {
    if docs.trim().is_empty() {
        return;
    }
    let text = format!(
        "# Library documentation (retrieved, for API/usage reference)\n\
         Current docs for libraries mentioned in the task — prefer them over prior knowledge for \
         API and usage details. For latest-version or other current-value questions, use the \
         'Current facts' block above instead (these docs may omit the version).\n\n{docs}"
    );
    append_system_block(req, text);
}

/// Inject relevant, still-fresh facts as a system block — time-stamped "as of" snapshots, so the
/// model treats them as current-as-of-that-date (push; streaming-preserving). No-op when empty.
pub fn inject_facts(req: &mut anthropic::AnthropicRequest, facts: &str) {
    if facts.trim().is_empty() {
        return;
    }
    let text = format!(
        "# Current facts (authoritative, time-stamped)\n\
         Verified current values for time-sensitive questions (latest versions, current state). \
         Use these directly as the answer — they OVERRIDE both your prior knowledge and the general \
         library documentation below. Each is stamped with when it was last confirmed; cite that \
         date, and suggest re-checking only if exact precision is critical.\n\n{facts}"
    );
    append_system_block(req, text);
}

/// Append one `text` system block after the client's existing system content (so the stable
/// client prefix is undisturbed for prefix caching).
fn append_system_block(req: &mut anthropic::AnthropicRequest, text: String) {
    let block = anthropic::SystemMessage {
        message_type: "text".to_string(),
        text,
        cache_control: None,
    };
    let new_system = match req.system.take() {
        None => anthropic::SystemPrompt::Multiple(vec![block]),
        Some(anthropic::SystemPrompt::Single(s)) => anthropic::SystemPrompt::Multiple(vec![
            anthropic::SystemMessage {
                message_type: "text".to_string(),
                text: s,
                cache_control: None,
            },
            block,
        ]),
        Some(anthropic::SystemPrompt::Multiple(mut blocks)) => {
            blocks.push(block);
            anthropic::SystemPrompt::Multiple(blocks)
        }
    };
    req.system = Some(new_system);
}

/// The injection/learning query — but ONLY when the latest turn is a fresh user message carrying
/// real text. Returns `None` for tool-loop continuations (the last message is a `tool_result` with
/// no text) or assistant turns, so retrieval + injection are skipped on those steps. In a multi-step
/// tool loop the same question would otherwise be re-embedded and re-injected on every step (wasted
/// GPU embeds + repeated injected tokens); this fires only on genuinely new user turns.
pub fn fresh_user_query(req: &anthropic::AnthropicRequest) -> Option<String> {
    let last = req.messages.last()?;
    if last.role != "user" {
        return None;
    }
    let text = match &last.content {
        anthropic::MessageContent::Text(t) => t.clone(),
        anthropic::MessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                anthropic::ContentBlock::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };
    let text = text.trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}
