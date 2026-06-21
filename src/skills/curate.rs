//! Stage 4: curate the skill store so it stays small and high-signal.
//!  - retention: drop unverified candidates older than the retention window (they never earned
//!    promotion, so they're noise).
//!  - dedup: collapse near-identical entries (cosine >= threshold), keeping the higher-tier /
//!    newer one — store bloat dilutes retrieval and over-injection degrades quality.
//!
//! A periodic background loop; all operations best-effort.

use super::store;
use crate::config::Config;
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const SCROLL_LIMIT: u32 = 500;
/// Qwen3-Embedding-4B (2560-dim): genuine paraphrases of one lesson sit ~0.85-0.92 while distinct
/// lessons stay ≤0.84, so 0.86 collapses the paraphrase floods without an LLM gate. (The old 0.93
/// was a bge-m3 value left un-recalibrated when embeddings switched — it silently disabled dedup,
/// letting candidates explode with 20+ copies of the same lesson.)
const DEDUP_THRESHOLD: f32 = 0.86;

pub fn spawn(config: Arc<Config>, client: Client) {
    if !config.skills.learn {
        return;
    }
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(45)).await;
        let mut tick =
            tokio::time::interval(Duration::from_secs(config.skills.curate_interval_secs));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            run_once(&config, &client).await;
        }
    });
    tracing::info!("skills/curate: loop started");
}

async fn run_once(config: &Config, client: &Client) {
    let qc = store::QdrantClient::new(
        config.skills.qdrant_url.clone(),
        config.skills.collection.clone(),
        client.clone(),
    );

    // 1. retention: drop unverified candidates past the window.
    let now = unix_now();
    let max_age = config.skills.retention_days as u64 * 86_400;
    let mut dropped = 0;
    for (id, p) in qc.scroll_tier("candidate", SCROLL_LIMIT).await {
        if let Some(created) = p.created_at {
            if now.saturating_sub(created) > max_age && qc.delete(id).await {
                dropped += 1;
            }
        }
    }
    if dropped > 0 {
        tracing::info!(dropped, "skills/curate: retention dropped stale candidates");
    }

    // 2. dedup: collapse near-identical entries, keeping the higher-priority one.
    let all = qc.scroll_all_with_vectors(SCROLL_LIMIT).await;
    let mut deleted: Vec<u64> = Vec::new();
    for i in 0..all.len() {
        if deleted.contains(&all[i].0) {
            continue;
        }
        for j in (i + 1)..all.len() {
            if deleted.contains(&all[j].0) {
                continue;
            }
            if cosine(&all[i].2, &all[j].2) >= DEDUP_THRESHOLD {
                let drop_id = if priority(&all[j].1) > priority(&all[i].1) {
                    all[i].0
                } else {
                    all[j].0
                };
                if qc.delete(drop_id).await {
                    deleted.push(drop_id);
                }
            }
        }
    }
    if !deleted.is_empty() {
        tracing::info!(removed = deleted.len(), "skills/curate: dedup removed near-duplicates");
    }
    if dropped > 0 || !deleted.is_empty() {
        super::eventlog::record("curate", json!({"dropped": dropped, "deduped": deleted.len()}));
    }
}

/// Keep-vs-drop priority for a duplicate pair (higher wins): trusted > verified > candidate, then
/// the more recently promoted/created.
fn priority(p: &store::SkillPayload) -> u64 {
    let tier = match p.tier.as_str() {
        "trusted" => 3,
        "verified" => 2,
        _ => 1,
    };
    tier * 10_000_000_000 + p.verified_at.or(p.created_at).unwrap_or(0)
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

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
