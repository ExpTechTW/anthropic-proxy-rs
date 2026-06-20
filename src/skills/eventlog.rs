//! Compact, persistent JSONL event log for the learning pipeline.
//!
//! Records just enough to analyse the learning funnel (distil → reject/verify → trust → inject)
//! and tune it — one small JSON object per line, pruned to a retention window so it never grows
//! without bound. Best-effort and off the request path: `record` only pushes a line into an
//! in-memory buffer (bounded, dropped if full); a background task batches it to disk every couple
//! of seconds and prunes hourly. Disabled unless a path is configured.
//!
//! Analyse with `jq`, e.g.:
//!   - funnel counts:   `jq -r .ev events.jsonl | sort | uniq -c`
//!   - what got learned: `jq -r 'select(.ev=="distill") | .skills[]' events.jsonl`
//!   - most-injected:    `jq -r 'select(.ev=="inject") | .skills[]' events.jsonl | sort | uniq -c | sort -rn`

use serde_json::{json, Value};
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// (path, retention_secs) — set once when enabled.
static CFG: OnceLock<(String, u64)> = OnceLock::new();
/// Pending lines awaiting a flush.
const MAX_PENDING: usize = 10_000;

fn buffer() -> &'static Mutex<Vec<String>> {
    static B: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    B.get_or_init(|| Mutex::new(Vec::new()))
}

/// Enable the event log (no-op if `path` is empty). Spawns the flusher + pruner.
pub fn init(path: &str, retention_days: u64) {
    if path.trim().is_empty() {
        return;
    }
    if CFG
        .set((path.to_string(), retention_days.max(1) * 86_400))
        .is_err()
    {
        return;
    }
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(2));
        loop {
            tick.tick().await;
            flush();
        }
    });
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(3600));
        loop {
            tick.tick().await;
            prune();
        }
    });
    if let Some((p, r)) = CFG.get() {
        tracing::info!(path = %p, retention_days = r / 86_400, "skills/eventlog: enabled");
    }
}

/// Record one event (non-blocking, best-effort). `fields` is a JSON object merged with `t`/`ev`.
pub fn record(event: &str, fields: Value) {
    if CFG.get().is_none() {
        return;
    }
    let mut obj = match fields {
        Value::Object(m) => m,
        _ => serde_json::Map::new(),
    };
    obj.insert("t".into(), json!(now()));
    obj.insert("ev".into(), json!(event));
    if let Ok(line) = serde_json::to_string(&Value::Object(obj)) {
        let mut b = buffer().lock().unwrap();
        if b.len() < MAX_PENDING {
            b.push(line);
        }
    }
}

fn flush() {
    let Some((path, _)) = CFG.get() else {
        return;
    };
    let lines = {
        let mut b = buffer().lock().unwrap();
        std::mem::take(&mut *b)
    };
    if lines.is_empty() {
        return;
    }
    if let Some(dir) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        for l in lines {
            let _ = writeln!(f, "{l}");
        }
    }
}

fn prune() {
    let Some((path, retention)) = CFG.get() else {
        return;
    };
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    let cutoff = now().saturating_sub(*retention);
    let mut kept = Vec::new();
    let mut changed = false;
    for l in content.lines() {
        let keep = serde_json::from_str::<Value>(l)
            .ok()
            .and_then(|v| v.get("t").and_then(Value::as_u64))
            .map(|t| t >= cutoff)
            .unwrap_or(true); // keep unparseable lines
        if keep {
            kept.push(l);
        } else {
            changed = true;
        }
    }
    if changed {
        let mut out = kept.join("\n");
        if !out.is_empty() {
            out.push('\n');
        }
        let _ = std::fs::write(path, out);
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
