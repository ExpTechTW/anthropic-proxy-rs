//! Minimal Qdrant REST client for the learned-skills store.
//!
//! We talk to Qdrant over its plain HTTP API with the proxy's shared `reqwest` client, so the
//! whole skills feature adds **zero new crates** (keeping `cargo build --locked` valid without a
//! lockfile regen). Only the handful of operations the feature needs are implemented:
//!   - `search`            — vector kNN with a `tier` filter + score threshold (Stage 1, read path)
//!   - `ensure_collection` — idempotent create with the embedding's dimension (Stage 2, write path)
//!   - `upsert`            — write one skill point (Stage 2+)
//!
//! Every method is best-effort: a missing collection, an unreachable Qdrant, or a malformed
//! response yields `None`/`false` (logged), never an error that could break a proxied request.

use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;

/// Qdrant is co-located (same docker network); a search/create that hangs must fail fast so the
/// request path the read-loop sits on never stalls.
const QDRANT_TIMEOUT: Duration = Duration::from_secs(15);

/// The stored fields of a skill point we read back on retrieval. Extra payload keys
/// (provenance, timestamps, usage counters added by later stages) are ignored here.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SkillPayload {
    // Filtering on `tier` is done Qdrant-side (scroll/search), so the field isn't read in Rust.
    #[serde(default)]
    #[allow(dead_code)]
    pub tier: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub when_to_use: String,
    #[serde(default)]
    pub body: String,
    /// When the entry was promoted to `verified` (unix secs) — drives the soak before `trusted`.
    #[serde(default)]
    pub verified_at: Option<u64>,
    /// When the entry was first written (unix secs) — drives retention/decay (Stage 4).
    #[serde(default)]
    #[allow(dead_code)]
    pub created_at: Option<u64>,
}

/// One search hit: the point id (as a string, whether Qdrant returned an int or uuid), its
/// cosine score, and the decoded payload.
#[derive(Debug, Clone)]
pub struct ScoredSkill {
    pub id: String,
    pub score: f32,
    pub payload: SkillPayload,
}

#[derive(Deserialize)]
struct SearchResponse {
    #[serde(default)]
    result: Vec<RawPoint>,
}

#[derive(Deserialize)]
struct RawPoint {
    id: Value,
    #[serde(default)]
    score: f32,
    #[serde(default)]
    payload: SkillPayload,
}

#[derive(Deserialize)]
struct ScrollResponse {
    result: ScrollResult,
}

#[derive(Deserialize)]
struct ScrollResult {
    #[serde(default)]
    points: Vec<ScrollPoint>,
}

#[derive(Deserialize)]
struct ScrollPoint {
    id: Value,
    #[serde(default)]
    payload: SkillPayload,
}

/// Decode a Qdrant point id (we always write u64 ids) back to u64.
fn id_as_u64(v: &Value) -> Option<u64> {
    match v {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// A client bound to one Qdrant base URL + collection.
pub struct QdrantClient {
    base: String,
    collection: String,
    http: Client,
}

impl QdrantClient {
    pub fn new(base: impl Into<String>, collection: impl Into<String>, http: Client) -> Self {
        Self {
            base: base.into().trim_end_matches('/').to_string(),
            collection: collection.into(),
            http,
        }
    }

    /// Vector kNN search restricted to the given tiers (OR), above `min_score`. Returns `None`
    /// on any failure — notably a 404 when the collection hasn't been created yet, which the
    /// caller treats as "no skills" (the common cold-start case).
    pub async fn search(
        &self,
        vector: &[f32],
        top_k: u32,
        min_score: f32,
        tiers: &[String],
    ) -> Option<Vec<ScoredSkill>> {
        let url = format!("{}/collections/{}/points/search", self.base, self.collection);
        let mut body = json!({
            "vector": vector,
            "limit": top_k,
            "with_payload": true,
            "score_threshold": min_score,
        });
        if !tiers.is_empty() {
            // `should` with one match-per-tier = OR (Qdrant defaults min_should to 1).
            let should: Vec<Value> = tiers
                .iter()
                .map(|t| json!({"key": "tier", "match": {"value": t}}))
                .collect();
            body["filter"] = json!({ "should": should });
        }

        let resp = self
            .http
            .post(&url)
            .timeout(QDRANT_TIMEOUT)
            .json(&body)
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            tracing::debug!("skills: qdrant search returned {}", resp.status());
            return None;
        }
        let parsed: SearchResponse = resp.json().await.ok()?;
        Some(
            parsed
                .result
                .into_iter()
                .map(|p| ScoredSkill {
                    id: point_id_string(&p.id),
                    score: p.score,
                    payload: p.payload,
                })
                .collect(),
        )
    }

    /// Scroll up to `limit` points in one tier (no vectors). Used by the verify/curate loops.
    /// Best-effort: empty on failure (e.g. collection not yet created).
    pub async fn scroll_tier(&self, tier: &str, limit: u32) -> Vec<(u64, SkillPayload)> {
        let url = format!("{}/collections/{}/points/scroll", self.base, self.collection);
        let body = json!({
            "limit": limit,
            "with_payload": true,
            "with_vector": false,
            "filter": { "must": [ { "key": "tier", "match": { "value": tier } } ] },
        });
        let Ok(resp) = self
            .http
            .post(&url)
            .timeout(QDRANT_TIMEOUT)
            .json(&body)
            .send()
            .await
        else {
            return Vec::new();
        };
        if !resp.status().is_success() {
            return Vec::new();
        }
        let Ok(parsed) = resp.json::<ScrollResponse>().await else {
            return Vec::new();
        };
        parsed
            .result
            .points
            .into_iter()
            .filter_map(|p| id_as_u64(&p.id).map(|id| (id, p.payload)))
            .collect()
    }

    /// Merge `fields` into a point's payload (Qdrant set_payload) without touching its vector —
    /// used to change `tier` and stamp verification metadata.
    pub async fn set_payload(&self, id: u64, fields: Value) -> bool {
        let url = format!("{}/collections/{}/points/payload?wait=true", self.base, self.collection);
        let body = json!({ "payload": fields, "points": [id] });
        matches!(
            self.http.post(&url).timeout(QDRANT_TIMEOUT).json(&body).send().await,
            Ok(r) if r.status().is_success()
        )
    }

    /// Delete one point by id (used by curation/retention).
    #[allow(dead_code)]
    pub async fn delete(&self, id: u64) -> bool {
        let url = format!("{}/collections/{}/points/delete?wait=true", self.base, self.collection);
        let body = json!({ "points": [id] });
        matches!(
            self.http.post(&url).timeout(QDRANT_TIMEOUT).json(&body).send().await,
            Ok(r) if r.status().is_success()
        )
    }

    /// Idempotently create the collection sized for the embedding model. No-op if it already
    /// exists. Used by the write path (Stage 2) once the first embedding reveals the dimension.
    #[allow(dead_code)]
    pub async fn ensure_collection(&self, size: usize) -> bool {
        let url = format!("{}/collections/{}", self.base, self.collection);
        if let Ok(resp) = self.http.get(&url).timeout(QDRANT_TIMEOUT).send().await {
            if resp.status().is_success() {
                return true;
            }
        }
        let body = json!({ "vectors": { "size": size, "distance": "Cosine" } });
        match self
            .http
            .put(&url)
            .timeout(QDRANT_TIMEOUT)
            .json(&body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => true,
            Ok(resp) => {
                tracing::warn!("skills: qdrant create collection returned {}", resp.status());
                false
            }
            Err(e) => {
                tracing::warn!("skills: qdrant create collection failed: {e}");
                false
            }
        }
    }

    /// Upsert one skill point (waits for the write to be applied). `payload` is the full skill
    /// document (tier, title, when_to_use, body, provenance, timestamps, …).
    #[allow(dead_code)]
    pub async fn upsert(&self, id: u64, vector: &[f32], payload: Value) -> bool {
        let url = format!("{}/collections/{}/points?wait=true", self.base, self.collection);
        let body = json!({ "points": [ { "id": id, "vector": vector, "payload": payload } ] });
        match self
            .http
            .put(&url)
            .timeout(QDRANT_TIMEOUT)
            .json(&body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => true,
            Ok(resp) => {
                tracing::warn!("skills: qdrant upsert returned {}", resp.status());
                false
            }
            Err(e) => {
                tracing::warn!("skills: qdrant upsert failed: {e}");
                false
            }
        }
    }
}

/// Render a Qdrant point id (int or string/uuid) as a string for logging and the
/// `x-injected-skills` header.
fn point_id_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// A deterministic u64 point id from a skill's identity text (e.g. title), so re-distilling the
/// same lesson updates the existing point instead of creating a duplicate.
#[allow(dead_code)]
pub fn stable_id(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}
