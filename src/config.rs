use anyhow::{bail, Result};
use reqwest::Url;
use std::{collections::BTreeMap, env, path::PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub bind: String,
    pub upstream_urls: Vec<String>,
    pub api_key: Option<String>,
    pub passthrough_api_key: bool,
    /// Forward count_tokens to the upstream `/tokenize` for exact counts (opt-in).
    pub upstream_tokenize: bool,
    pub model_map: BTreeMap<String, String>,
    /// Maps a thinking budget to an upstream `reasoning_effort` (global tiers + per-model overrides).
    pub effort_map: EffortMap,
    pub system_prompt_ignore_terms: Vec<String>,
    pub reasoning_model: Option<String>,
    pub completion_model: Option<String>,
    pub debug: bool,
    pub verbose: bool,
    /// Log every request's fields (minus `messages`/`system`) at INFO, so new/unknown
    /// client fields are visible for debugging without dumping message bodies.
    pub log_requests: bool,
    /// Seconds of streaming-output silence before emitting an SSE keep-alive comment,
    /// so a fronting proxy (e.g. Cloudflare's free plan, 100s) doesn't abort the stream
    /// during the gap before the first token. 0 disables the heartbeat.
    pub heartbeat_secs: u64,
    /// MCP (Streamable HTTP) endpoint of the co-located `open-websearch` server, used to
    /// emulate Anthropic's server-side `web_search` / `web_fetch` tools for models that
    /// can't browse. Defaults to the in-container instance started by docker-entrypoint.sh.
    pub websearch_url: String,
    /// The model the web-search/fetch agent loop routes its backend calls through (its "search
    /// agent id", e.g. `auto`). `None` (env unset) disables emulation entirely: the web tools
    /// are stripped and the client gets an empty result instead of a real search.
    pub websearch_model: Option<String>,
    /// Base URL of a self-hosted SearXNG instance (e.g. `http://searxng:8080`). When set, the
    /// web-search agent runs `web_search` through SearXNG's JSON API (70+ engines, deduped)
    /// instead of open-websearch; `web_fetch` still uses open-websearch. `None` → open-websearch.
    pub searxng_url: Option<String>,
    /// Self-learning skill-injection settings (read path; see [`crate::skills`]).
    pub skills: SkillsConfig,
}

/// Configuration for auto-injecting learned skills into requests. Disabled by default — the whole
/// feature is gated behind `ANTHROPIC_PROXY_SKILLS_ENABLED` so default behaviour is unchanged.
#[derive(Debug, Clone)]
pub struct SkillsConfig {
    /// Master switch. When false, the proxy never embeds, queries Qdrant, or injects.
    pub enabled: bool,
    /// Qdrant base URL (co-located service), e.g. `http://qdrant:6333`.
    pub qdrant_url: String,
    /// Qdrant collection holding the skill points.
    pub collection: String,
    /// Separate Qdrant collection for the factual-memory store (time-sensitive facts; see
    /// [`crate::skills`]). Kept apart from skills — facts have a different lifecycle (they decay).
    pub facts_collection: String,
    /// Explicit embeddings endpoint; `None` derives one from the first upstream chat URL.
    pub embed_url: Option<String>,
    /// Embedding model name sent to the embeddings endpoint. Empty disables retrieval.
    pub embed_model: String,
    /// Max skills injected per request — capped low; over-injection degrades quality.
    pub top_k: u32,
    /// Minimum cosine score for a skill to be injected (filters weak/irrelevant matches).
    pub min_score: f32,
    /// Trust tiers eligible for injection (e.g. `verified`, `trusted`). Candidates are excluded.
    pub inject_tiers: Vec<String>,
    /// Learn from finished conversations (Stage 2 distillation write path).
    pub learn: bool,
    /// Model for background learning LLM calls (distil/judge/verify). Default `auto`.
    pub llm_model: String,
    /// API key for background LLM/embedding calls not tied to a client request (distillation,
    /// verification, proactive study). When `llm_url` is set this key is sent to it as a bearer;
    /// otherwise it authes the upstream fallback. Falls back to the last-seen client key when unset.
    pub api_key: Option<String>,
    /// Retention for unverified candidates (days) before curation drops them.
    pub retention_days: u32,
    /// Chat endpoint for background learning LLM calls. When set, calls go here with `api_key` as a
    /// bearer if that is set (e.g. an authed/metered upstream), else with no auth (e.g. a no-auth
    /// internal backend); unset uses the authed upstream + the client key.
    pub llm_url: Option<String>,
    /// How often the verification/promotion loop runs (seconds).
    pub verify_interval_secs: u64,
    /// How long a `verified` entry must soak before it can become `trusted` (seconds).
    pub soak_secs: u64,
    /// Don't re-attempt verifying a candidate that recently failed corroboration, for this many
    /// seconds (avoids re-searching the same un-promotable candidate every cycle).
    pub verify_backoff_secs: u64,
    /// How often the curation loop (retention + dedup) runs (seconds).
    pub curate_interval_secs: u64,
    /// Enable proactive learning (research recent asked questions into candidates).
    pub proactive: bool,
    /// Cache corroborated time-sensitive facts (versions, current state) into the facts store,
    /// each stamped with an observation time + volatility-derived half-life.
    pub facts: bool,
    /// How often the proactive-learning loop runs (seconds).
    pub proactive_interval_secs: u64,
    /// How often the facts validity loop re-checks decayed facts (seconds).
    pub facts_validity_interval_secs: u64,
    /// Path to a compact JSONL learning-event log (empty disables it). Persist via a volume.
    pub eventlog_path: String,
    /// Days to retain learning-event log entries.
    pub eventlog_retention_days: u64,
    /// Inject proxy-handled tools (`recall_skills` + `search_docs`) the model can call on demand.
    /// Requests then run through a tool loop (buffered + replayed; no token streaming).
    pub tools: bool,
    /// docs-mcp MCP endpoint for the `search_docs` tool (None disables that tool).
    pub docs_mcp_url: Option<String>,
    /// Push-inject docs from docs-mcp for indexed libraries mentioned in a request (streaming-safe,
    /// unlike the tool loop). Needs `docs_mcp_url`.
    pub docs_inject: bool,
}

impl Default for SkillsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            qdrant_url: "http://qdrant:6333".to_string(),
            collection: "skills".to_string(),
            facts_collection: "facts".to_string(),
            embed_url: None,
            embed_model: String::new(),
            top_k: 3,
            min_score: 0.5,
            inject_tiers: vec!["verified".to_string(), "trusted".to_string()],
            learn: false,
            llm_model: "auto".to_string(),
            api_key: None,
            retention_days: 30,
            llm_url: None,
            verify_interval_secs: 300,
            soak_secs: 14 * 24 * 60 * 60,
            verify_backoff_secs: 6 * 60 * 60,
            curate_interval_secs: 600,
            proactive: false,
            facts: false,
            proactive_interval_secs: 600,
            facts_validity_interval_secs: 1800,
            eventlog_path: String::new(),
            eventlog_retention_days: 7,
            tools: false,
            docs_mcp_url: None,
            docs_inject: false,
        }
    }
}

impl SkillsConfig {
    fn from_env() -> Self {
        let d = SkillsConfig::default();
        let enabled = env::var("ANTHROPIC_PROXY_SKILLS_ENABLED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let qdrant_url = env::var("ANTHROPIC_PROXY_SKILLS_QDRANT_URL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or(d.qdrant_url);
        let collection = env::var("ANTHROPIC_PROXY_SKILLS_COLLECTION")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or(d.collection);
        let facts_collection = env::var("ANTHROPIC_PROXY_SKILLS_FACTS_COLLECTION")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or(d.facts_collection);
        let embed_url = env::var("ANTHROPIC_PROXY_SKILLS_EMBED_URL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        let embed_model = env::var("ANTHROPIC_PROXY_SKILLS_EMBED_MODEL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_default();
        let top_k = env::var("ANTHROPIC_PROXY_SKILLS_TOP_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.top_k);
        let min_score = env::var("ANTHROPIC_PROXY_SKILLS_MIN_SCORE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.min_score);
        let inject_tiers = env::var("ANTHROPIC_PROXY_SKILLS_INJECT_TIERS")
            .ok()
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or(d.inject_tiers);
        let learn = env::var("ANTHROPIC_PROXY_SKILLS_LEARN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let llm_model = env::var("ANTHROPIC_PROXY_SKILLS_LLM_MODEL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or(d.llm_model);
        let api_key = env::var("ANTHROPIC_PROXY_SKILLS_API_KEY")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        let retention_days = env::var("ANTHROPIC_PROXY_SKILLS_RETENTION_DAYS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.retention_days);
        let llm_url = env::var("ANTHROPIC_PROXY_SKILLS_LLM_URL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        let verify_interval_secs = env::var("ANTHROPIC_PROXY_SKILLS_VERIFY_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.verify_interval_secs);
        let soak_secs = env::var("ANTHROPIC_PROXY_SKILLS_SOAK_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.soak_secs);
        let verify_backoff_secs = env::var("ANTHROPIC_PROXY_SKILLS_VERIFY_BACKOFF_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.verify_backoff_secs);
        let curate_interval_secs = env::var("ANTHROPIC_PROXY_SKILLS_CURATE_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.curate_interval_secs);
        let proactive = env::var("ANTHROPIC_PROXY_SKILLS_PROACTIVE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let facts = env::var("ANTHROPIC_PROXY_SKILLS_FACTS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let proactive_interval_secs = env::var("ANTHROPIC_PROXY_SKILLS_PROACTIVE_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.proactive_interval_secs);
        let facts_validity_interval_secs = env::var("ANTHROPIC_PROXY_SKILLS_FACTS_VALIDITY_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.facts_validity_interval_secs);
        let eventlog_path = env::var("ANTHROPIC_PROXY_SKILLS_EVENTLOG_PATH")
            .ok()
            .map(|v| v.trim().to_string())
            .unwrap_or_default();
        let eventlog_retention_days = env::var("ANTHROPIC_PROXY_SKILLS_EVENTLOG_RETENTION_DAYS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.eventlog_retention_days);
        let tools = env::var("ANTHROPIC_PROXY_SKILLS_TOOLS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let docs_mcp_url = env::var("ANTHROPIC_PROXY_SKILLS_DOCS_MCP_URL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        let docs_inject = env::var("ANTHROPIC_PROXY_SKILLS_DOCS_INJECT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        SkillsConfig {
            enabled,
            qdrant_url,
            collection,
            embed_url,
            embed_model,
            top_k,
            min_score,
            inject_tiers,
            learn,
            llm_model,
            api_key,
            retention_days,
            llm_url,
            verify_interval_secs,
            soak_secs,
            verify_backoff_secs,
            curate_interval_secs,
            proactive,
            facts,
            facts_collection,
            proactive_interval_secs,
            facts_validity_interval_secs,
            eventlog_path,
            eventlog_retention_days,
            tools,
            docs_mcp_url,
            docs_inject,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: 3000,
            bind: "0.0.0.0".to_string(),
            upstream_urls: vec!["http://localhost:11434".to_string()],
            api_key: None,
            passthrough_api_key: false,
            upstream_tokenize: false,
            model_map: BTreeMap::new(),
            effort_map: EffortMap::default(),
            system_prompt_ignore_terms: Vec::new(),
            reasoning_model: None,
            completion_model: None,
            debug: false,
            verbose: false,
            log_requests: false,
            heartbeat_secs: 15,
            websearch_url: "http://localhost:3100/mcp".to_string(),
            websearch_model: None,
            searxng_url: None,
            skills: SkillsConfig::default(),
        }
    }
}

/// Upper bound on the thinking budget we map (matches a typical 16K max output).
const MAX_THINKING_BUDGET: u32 = 16384;

/// Maps a thinking `budget_tokens` to an upstream `reasoning_effort`: a global tier list plus
/// optional per-(client-)model overrides.
#[derive(Debug, Clone, Default)]
pub struct EffortMap {
    /// (max_budget, effort) sorted ascending — used for models without an override.
    global: Vec<(u32, String)>,
    /// Per-model tier lists (keyed on the client model name).
    overrides: BTreeMap<String, Vec<(u32, String)>>,
}

impl EffortMap {
    /// Parse `"low:2048,medium:8192,high:16384;claude-haiku-3-5=low:512,medium:4096,high:16384"`.
    /// A `;`-segment without `=` sets the **global** tiers; `model=tiers` segments **override**
    /// a specific client model. Each tier is `effort:maxBudget`; tiers are sorted by budget.
    pub fn parse(spec: &str) -> Self {
        let mut map = EffortMap::default();
        for segment in spec.split(';') {
            let segment = segment.trim();
            if segment.is_empty() {
                continue;
            }
            match segment.split_once('=') {
                Some((model, tiers)) => {
                    let tiers = parse_tiers(tiers);
                    if !tiers.is_empty() {
                        map.overrides.insert(model.trim().to_string(), tiers);
                    }
                }
                None => {
                    let tiers = parse_tiers(segment);
                    if !tiers.is_empty() {
                        map.global = tiers;
                    }
                }
            }
        }
        map
    }

    pub fn is_empty(&self) -> bool {
        self.global.is_empty() && self.overrides.is_empty()
    }

    /// Resolve a `reasoning_effort` for `model` and a thinking `budget` (tokens), or `None`
    /// when no tiers apply. The budget is clamped to [`MAX_THINKING_BUDGET`]; the chosen tier
    /// is the first whose `maxBudget >= budget`, or the highest tier when it exceeds them all.
    pub fn resolve(&self, model: &str, budget: u32) -> Option<String> {
        let tiers = match self.overrides.get(model) {
            Some(tiers) if !tiers.is_empty() => tiers,
            _ => &self.global,
        };
        let budget = budget.min(MAX_THINKING_BUDGET);
        tiers
            .iter()
            .find(|(max_budget, _)| budget <= *max_budget)
            .or_else(|| tiers.last())
            .map(|(_, effort)| effort.clone())
    }
}

/// Parse `"low:2048,medium:8192,high:16384"` into ascending `(max_budget, effort)` tiers.
/// Effort labels are passed through as-is (lowercased) — validating them is the upstream's job.
fn parse_tiers(spec: &str) -> Vec<(u32, String)> {
    let mut tiers: Vec<(u32, String)> = spec
        .split(',')
        .filter_map(|pair| {
            let (effort, budget) = pair.trim().split_once(':')?;
            let budget: u32 = budget.trim().parse().ok()?;
            Some((budget, effort.trim().to_ascii_lowercase()))
        })
        .collect();
    tiers.sort_by_key(|(budget, _)| *budget);
    tiers
}

impl Config {
    fn load_dotenv(custom_path: Option<PathBuf>) -> Option<PathBuf> {
        if let Some(path) = custom_path {
            if path.exists() && dotenvy::from_path(&path).is_ok() {
                return Some(path);
            }
            eprintln!(
                "⚠️  WARNING: Custom config file not found: {}",
                path.display()
            );
        }

        if let Ok(path) = dotenvy::dotenv() {
            return Some(path);
        }

        if let Ok(home) = env::var("HOME") {
            let home_config = PathBuf::from(home).join(".anthropic-proxy.env");
            if home_config.exists() && dotenvy::from_path(&home_config).is_ok() {
                return Some(home_config);
            }
        }

        let etc_config = PathBuf::from("/etc/anthropic-proxy/.env");
        if etc_config.exists() && dotenvy::from_path(&etc_config).is_ok() {
            return Some(etc_config);
        }

        None
    }

    pub fn from_env_with_path(custom_path: Option<PathBuf>) -> Result<Self> {
        if let Some(path) = Self::load_dotenv(custom_path) {
            eprintln!("📄 Loaded config from: {}", path.display());
        } else {
            eprintln!("ℹ️  No .env file found, using environment variables only");
        }

        let port = env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(3000);

        let bind = env::var("ANTHROPIC_PROXY_BIND")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "0.0.0.0".to_string());

        let raw_urls = env::var("UPSTREAM_BASE_URL")
            .or_else(|_| env::var("ANTHROPIC_PROXY_BASE_URL"))
            .map_err(|_| {
                anyhow::anyhow!(
                    "UPSTREAM_BASE_URL is required. Set it to your OpenAI-compatible endpoint.\n\
                Examples:\n\
                  - OpenRouter: https://openrouter.ai/api\n\
                  - OpenAI: https://api.openai.com\n\
                  - Multiple (failover): https://openrouter.ai/api;https://api.openai.com\n\
                  - Local: http://localhost:11434"
                )
            })?;

        let upstream_urls = Self::parse_upstream_urls(&raw_urls)?;

        let api_key = env::var("UPSTREAM_API_KEY")
            .or_else(|_| env::var("OPENROUTER_API_KEY"))
            .ok()
            .filter(|k| !k.is_empty());

        let model_map = env::var("ANTHROPIC_PROXY_MODEL_MAP")
            .ok()
            .map(|value| Self::parse_model_map(&value))
            .transpose()?
            .unwrap_or_default();

        let effort_map = env::var("ANTHROPIC_PROXY_EFFORT_MAP")
            .ok()
            .map(|value| EffortMap::parse(&value))
            .unwrap_or_default();

        let mut system_prompt_ignore_terms = env::var("ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS")
            .ok()
            .map(|value| Self::parse_system_prompt_ignore_terms(&value))
            .unwrap_or_default();
        Self::dedupe_ignore_terms(&mut system_prompt_ignore_terms);

        let reasoning_model = env::var("REASONING_MODEL").ok();
        let completion_model = env::var("COMPLETION_MODEL").ok();

        let debug = env::var("DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let verbose = env::var("VERBOSE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let passthrough_api_key = env::var("UPSTREAM_API_KEY_PASSTHROUGH")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let log_requests = env::var("ANTHROPIC_PROXY_LOG_REQUESTS")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let upstream_tokenize = env::var("ANTHROPIC_PROXY_UPSTREAM_TOKENIZE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let heartbeat_secs = env::var("ANTHROPIC_PROXY_HEARTBEAT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(15);

        let websearch_url = env::var("ANTHROPIC_PROXY_WEBSEARCH_URL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "http://localhost:3100/mcp".to_string());

        let websearch_model = env::var("ANTHROPIC_PROXY_WEBSEARCH_MODEL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());

        let searxng_url = env::var("ANTHROPIC_PROXY_SEARXNG_URL")
            .ok()
            .map(|v| v.trim().trim_end_matches('/').to_string())
            .filter(|v| !v.is_empty());

        let skills = SkillsConfig::from_env();

        // Validate: UPSTREAM_API_KEY_PASSTHROUGH requires UPSTREAM_API_KEY to be unset
        if passthrough_api_key && api_key.is_some() {
            bail!(
                "UPSTREAM_API_KEY_PASSTHROUGH=true cannot be used together with UPSTREAM_API_KEY.\n\
                 When passthrough is enabled, the API key is extracted from each incoming request's x-api-key header.\n\
                 Unset UPSTREAM_API_KEY or set UPSTREAM_API_KEY_PASSTHROUGH=false."
            );
        }

        Ok(Config {
            port,
            bind,
            upstream_urls,
            api_key,
            passthrough_api_key,
            upstream_tokenize,
            model_map,
            effort_map,
            system_prompt_ignore_terms,
            reasoning_model,
            completion_model,
            debug,
            verbose,
            log_requests,
            heartbeat_secs,
            websearch_url,
            websearch_model,
            searxng_url,
            skills,
        })
    }

    /// Upstream `/tokenize` URLs (siblings of the chat-completions endpoints).
    pub fn tokenize_urls(&self) -> Vec<String> {
        self.chat_completions_urls()
            .into_iter()
            .map(|url| url.replace("/chat/completions", "/tokenize"))
            .collect()
    }

    pub fn chat_completions_urls(&self) -> Vec<String> {
        self.upstream_urls
            .iter()
            .map(|url| {
                Self::resolve_chat_completions_url(url)
                    .expect("URLs should be validated during configuration loading")
            })
            .collect()
    }

    /// Upstream `/embeddings` URLs (siblings of the chat-completions endpoints), used for
    /// skill-retrieval embeddings when no explicit `ANTHROPIC_PROXY_SKILLS_EMBED_URL` is set.
    pub fn embeddings_urls(&self) -> Vec<String> {
        self.chat_completions_urls()
            .into_iter()
            .map(|url| url.replace("/chat/completions", "/embeddings"))
            .collect()
    }

    /// The embeddings endpoint for skill retrieval: the explicit override if set, else the first
    /// upstream-derived `/embeddings` URL. `None` only if there are somehow no upstreams.
    pub fn skills_embed_url(&self) -> Option<String> {
        if let Some(url) = &self.skills.embed_url {
            return Some(url.clone());
        }
        self.embeddings_urls().into_iter().next()
    }

    pub fn models_urls(&self) -> Vec<String> {
        self.upstream_urls
            .iter()
            .map(|url| {
                Self::resolve_models_url(url)
                    .expect("URLs should be validated during configuration loading")
            })
            .collect()
    }

    fn parse_upstream_urls(raw: &str) -> Result<Vec<String>> {
        let urls: Vec<String> = raw
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect();

        if urls.is_empty() {
            bail!("UPSTREAM_BASE_URL must not be empty");
        }

        for url in &urls {
            Self::resolve_chat_completions_url(url)?;
        }

        Ok(urls)
    }

    fn resolve_chat_completions_url(base_url: &str) -> Result<String> {
        let (normalized, path_segments) = Self::parse_base_url(base_url)?;

        if Self::is_chat_completions_path(&path_segments) {
            return Ok(normalized.to_string());
        }

        let last_segment = path_segments.last().map(String::as_str);
        if matches!(last_segment, Some("chat") | Some("completions")) {
            bail!(
                "UPSTREAM_BASE_URL must be either a service base URL, a versioned base URL like https://gateway.example.com/v2, or the full .../chat/completions endpoint"
            );
        }

        if last_segment.is_some_and(Self::is_version_segment) {
            return Ok(format!("{}/chat/completions", normalized));
        }

        Ok(format!("{}/v1/chat/completions", normalized))
    }

    fn resolve_models_url(base_url: &str) -> Result<String> {
        let (normalized, path_segments) = Self::parse_base_url(base_url)?;

        if Self::is_chat_completions_path(&path_segments) {
            let base = normalized
                .trim_end_matches("/chat/completions")
                .trim_end_matches('/');
            return Ok(format!("{}/models", base));
        }

        let last_segment = path_segments.last().map(String::as_str);
        if matches!(last_segment, Some("chat") | Some("completions")) {
            bail!(
                "UPSTREAM_BASE_URL must be either a service base URL, a versioned base URL like https://gateway.example.com/v2, or the full .../chat/completions endpoint"
            );
        }

        if last_segment.is_some_and(Self::is_version_segment) {
            return Ok(format!("{}/models", normalized));
        }

        Ok(format!("{}/v1/models", normalized))
    }

    fn parse_base_url(base_url: &str) -> Result<(String, Vec<String>)> {
        let normalized = base_url.trim();

        if normalized.is_empty() {
            bail!("UPSTREAM_BASE_URL must not be empty");
        }

        let parsed = Url::parse(normalized).map_err(|err| {
            anyhow::anyhow!("UPSTREAM_BASE_URL must be a valid http(s) URL: {}", err)
        })?;

        if !matches!(parsed.scheme(), "http" | "https") {
            bail!("UPSTREAM_BASE_URL must use http or https");
        }

        if parsed.query().is_some() || parsed.fragment().is_some() {
            bail!("UPSTREAM_BASE_URL must not include query parameters or fragments");
        }

        let path_segments: Vec<_> = parsed
            .path_segments()
            .map(|segments| {
                segments
                    .filter(|segment| !segment.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();

        Ok((normalized.trim_end_matches('/').to_string(), path_segments))
    }

    fn is_chat_completions_path(segments: &[String]) -> bool {
        matches!(segments, [.., chat, completions] if chat == "chat" && completions == "completions")
    }

    fn is_version_segment(segment: &str) -> bool {
        let version = segment
            .strip_prefix('v')
            .or_else(|| segment.strip_prefix('V'));

        version
            .is_some_and(|value| !value.is_empty() && value.chars().all(|ch| ch.is_ascii_digit()))
    }

    pub fn parse_system_prompt_ignore_terms(value: &str) -> Vec<String> {
        value
            .split([';', '\n'])
            .map(str::trim)
            .filter(|term| !term.is_empty())
            .map(ToOwned::to_owned)
            .collect()
    }

    pub fn dedupe_ignore_terms(terms: &mut Vec<String>) {
        let mut deduped = Vec::new();
        let mut seen = Vec::new();
        for term in terms.drain(..) {
            let normalized = term
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_ascii_lowercase();
            if !seen.iter().any(|existing: &String| existing == &normalized) {
                seen.push(normalized);
                deduped.push(term);
            }
        }
        *terms = deduped;
    }

    pub fn parse_model_map(value: &str) -> Result<BTreeMap<String, String>> {
        let mut model_map = BTreeMap::new();

        for entry in value
            .split([';', '\n'])
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
        {
            let (source, target) = entry.split_once('=').ok_or_else(|| {
                anyhow::anyhow!(
                    "Invalid ANTHROPIC_PROXY_MODEL_MAP entry '{}'. Expected source=target",
                    entry
                )
            })?;

            let source = source.trim();
            let target = target.trim();

            if source.is_empty() || target.is_empty() {
                bail!(
                    "Invalid ANTHROPIC_PROXY_MODEL_MAP entry '{}'. Source and target models must be non-empty",
                    entry
                );
            }

            model_map.insert(source.to_string(), target.to_string());
        }

        Ok(model_map)
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, EffortMap};

    #[test]
    fn effort_map_global_tiers_pick_by_budget() {
        let map = EffortMap::parse("low:2048,medium:8192,high:16384");
        assert_eq!(map.resolve("any", 500).as_deref(), Some("low"));
        assert_eq!(map.resolve("any", 2048).as_deref(), Some("low"));
        assert_eq!(map.resolve("any", 4000).as_deref(), Some("medium"));
        assert_eq!(map.resolve("any", 10000).as_deref(), Some("high"));
        // Above the top tier (and beyond the 16K clamp) → highest tier.
        assert_eq!(map.resolve("any", 31999).as_deref(), Some("high"));
    }

    #[test]
    fn effort_map_per_model_override_wins() {
        let map =
            EffortMap::parse("low:2048,medium:8192,high:16384;claude-haiku-3-5=low:512,high:16384");
        // Override model: 1000 > 512 → next tier (high).
        assert_eq!(
            map.resolve("claude-haiku-3-5", 1000).as_deref(),
            Some("high")
        );
        // Unlisted model falls back to the global tiers.
        assert_eq!(map.resolve("claude-opus-4-7", 1000).as_deref(), Some("low"));
    }

    #[test]
    fn effort_map_passes_labels_through_verbatim() {
        // Labels are forwarded as-is (lowercased); the upstream validates/defaults them.
        let map = EffortMap::parse("ULTRA:2048,high:16384");
        assert_eq!(map.resolve("any", 1000).as_deref(), Some("ultra"));
    }

    #[test]
    fn effort_map_empty_when_unset() {
        assert!(EffortMap::default().is_empty());
        assert_eq!(EffortMap::default().resolve("any", 5000), None);
    }

    #[test]
    fn base_url_without_version_defaults_to_v1_endpoint() {
        let url = Config::resolve_chat_completions_url("https://api.openai.com").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn versioned_base_url_preserves_existing_version() {
        let url = Config::resolve_chat_completions_url("https://gateway.example.com/v2").unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/chat/completions");
    }

    #[test]
    fn full_chat_completions_endpoint_is_used_as_is() {
        let url = Config::resolve_chat_completions_url(
            "https://gateway.example.com/v2/chat/completions/",
        )
        .unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/chat/completions");
    }

    #[test]
    fn models_url_without_version_defaults_to_v1_endpoint() {
        let url = Config::resolve_models_url("https://api.openai.com").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/models");
    }

    #[test]
    fn versioned_models_url_preserves_existing_version() {
        let url = Config::resolve_models_url("https://gateway.example.com/v2").unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/models");
    }

    #[test]
    fn full_chat_completions_endpoint_resolves_models_url() {
        let url =
            Config::resolve_models_url("https://gateway.example.com/v2/chat/completions").unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/models");
    }

    #[test]
    fn partial_chat_path_is_rejected() {
        let err = Config::resolve_chat_completions_url("https://gateway.example.com/v2/chat")
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("service base URL, a versioned base URL"));
    }

    #[test]
    fn query_strings_are_rejected() {
        let err = Config::resolve_chat_completions_url("https://gateway.example.com/v2?foo=bar")
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("must not include query parameters or fragments"));
    }

    #[test]
    fn fragments_are_rejected() {
        let err = Config::resolve_chat_completions_url("https://gateway.example.com/v2#section")
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("must not include query parameters or fragments"));
    }

    #[test]
    fn empty_url_is_rejected() {
        let err = Config::resolve_chat_completions_url("").unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn non_http_scheme_is_rejected() {
        let err = Config::resolve_chat_completions_url("ftp://gateway.example.com").unwrap_err();
        assert!(err.to_string().contains("must use http or https"));
    }

    #[test]
    fn explicit_v1_is_preserved_not_doubled() {
        let url = Config::resolve_chat_completions_url("https://openrouter.ai/api/v1").unwrap();
        assert_eq!(url, "https://openrouter.ai/api/v1/chat/completions");
    }

    #[test]
    fn trailing_slash_on_base_url_is_normalized() {
        let url = Config::resolve_chat_completions_url("https://api.openai.com/").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn models_url_from_explicit_v1() {
        let url = Config::resolve_models_url("https://openrouter.ai/api/v1").unwrap();
        assert_eq!(url, "https://openrouter.ai/api/v1/models");
    }

    #[test]
    fn models_url_with_trailing_slash() {
        let url = Config::resolve_models_url("https://api.openai.com/").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/models");
    }

    #[test]
    fn url_with_subpath_and_no_version_defaults_to_v1() {
        let url = Config::resolve_chat_completions_url("https://openrouter.ai/api").unwrap();
        assert_eq!(url, "https://openrouter.ai/api/v1/chat/completions");
    }

    #[test]
    fn only_completions_path_is_rejected() {
        let err =
            Config::resolve_chat_completions_url("https://gateway.example.com/v2/completions")
                .unwrap_err();
        assert!(err
            .to_string()
            .contains("service base URL, a versioned base URL"));
    }

    #[test]
    fn uppercase_version_prefix_is_accepted() {
        let url = Config::resolve_chat_completions_url("https://gateway.example.com/V2").unwrap();
        assert_eq!(url, "https://gateway.example.com/V2/chat/completions");
    }

    #[test]
    fn parse_system_prompt_ignore_terms_supports_semicolons_and_newlines() {
        let terms =
            Config::parse_system_prompt_ignore_terms("rm -rf;git reset --hard\nsudo rm -rf");

        assert_eq!(
            terms,
            vec![
                "rm -rf".to_string(),
                "git reset --hard".to_string(),
                "sudo rm -rf".to_string()
            ]
        );
    }

    #[test]
    fn dedupe_ignore_terms_normalizes_case_and_whitespace() {
        let mut terms = vec![
            "rm -rf".to_string(),
            " RM\t-rF ".to_string(),
            "git reset --hard".to_string(),
        ];

        Config::dedupe_ignore_terms(&mut terms);

        assert_eq!(
            terms,
            vec!["rm -rf".to_string(), "git reset --hard".to_string()]
        );
    }

    #[test]
    fn parse_model_map_supports_semicolons_and_newlines() {
        let model_map = Config::parse_model_map(
            "claude-3-5-sonnet=openai/gpt-5.2-chat\nclaude-haiku=openai/gpt-4.1-mini",
        )
        .unwrap();

        assert_eq!(
            model_map.get("claude-3-5-sonnet"),
            Some(&"openai/gpt-5.2-chat".to_string())
        );
        assert_eq!(
            model_map.get("claude-haiku"),
            Some(&"openai/gpt-4.1-mini".to_string())
        );
    }

    #[test]
    fn parse_model_map_rejects_invalid_entries() {
        let err = Config::parse_model_map("claude-3-5-sonnet").unwrap_err();

        assert!(err.to_string().contains("Expected source=target"));
    }

    #[test]
    fn parse_upstream_urls_splits_on_semicolons() {
        let urls = Config::parse_upstream_urls("https://openrouter.ai/api;https://api.openai.com")
            .unwrap();

        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://openrouter.ai/api");
        assert_eq!(urls[1], "https://api.openai.com");
    }

    #[test]
    fn parse_upstream_urls_single_url_still_works() {
        let urls = Config::parse_upstream_urls("https://api.openai.com").unwrap();
        assert_eq!(urls.len(), 1);
    }

    #[test]
    fn parse_upstream_urls_rejects_empty() {
        let err = Config::parse_upstream_urls("").unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn parse_upstream_urls_validates_each_url() {
        let err = Config::parse_upstream_urls("https://api.openai.com;not-a-url").unwrap_err();
        assert!(err.to_string().contains("valid http"));
    }

    #[test]
    fn chat_completions_urls_resolves_all() {
        let config = Config {
            upstream_urls: vec![
                "https://openrouter.ai/api".to_string(),
                "https://api.openai.com".to_string(),
            ],
            ..Default::default()
        };

        let urls = config.chat_completions_urls();
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://openrouter.ai/api/v1/chat/completions");
        assert_eq!(urls[1], "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn passthrough_api_key_defaults_to_false() {
        let config = Config::default();
        assert!(!config.passthrough_api_key);
    }

    #[test]
    fn passthrough_disabled_with_static_key_works() {
        let config = Config {
            api_key: Some("sk-test".to_string()),
            passthrough_api_key: false,
            ..Default::default()
        };
        assert!(!config.passthrough_api_key);
        assert_eq!(config.api_key, Some("sk-test".to_string()));
    }

    #[test]
    fn passthrough_enabled_with_no_static_key() {
        let config = Config {
            api_key: None,
            passthrough_api_key: true,
            ..Default::default()
        };
        assert!(config.passthrough_api_key);
        assert!(config.api_key.is_none());
    }

    #[test]
    fn bind_defaults_to_zero_zero_zero_zero() {
        let config = Config::default();
        assert_eq!(config.bind, "0.0.0.0");
    }

    #[test]
    fn bind_accepts_loopback() {
        let config = Config {
            bind: "127.0.0.1".to_string(),
            ..Default::default()
        };
        assert_eq!(config.bind, "127.0.0.1");
    }
}
