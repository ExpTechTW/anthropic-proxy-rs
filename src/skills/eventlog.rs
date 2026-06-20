//! Compact, persistent JSONL event log for the learning pipeline, rotated into **daily files**.
//!
//! Records just enough to analyse the learning funnel (distil → reject/verify → trust → inject)
//! and tune it — one small JSON object per line. Files are `<base>-YYYYMMDD.jsonl` (UTC date);
//! pruning simply deletes whole files older than the retention window (no rewrite, no race with
//! appends). Best-effort and off the request path: `record` only pushes a line into an in-memory
//! buffer (bounded, dropped if full); a background task batches it to today's file every couple of
//! seconds and prunes hourly. Disabled unless a path is configured.
//!
//! Analyse across days with a glob, e.g.:
//!   - funnel counts:    `jq -r .ev skills-events-*.jsonl | sort | uniq -c`
//!   - what got learned: `jq -r 'select(.ev=="distill")|.skills[]' skills-events-*.jsonl`
//!   - most-injected:    `jq -r 'select(.ev=="inject")|.skills[]' skills-events-*.jsonl | sort | uniq -c | sort -rn`

use serde_json::{json, Value};
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// (stem, retention_secs). `stem` is the configured path minus a trailing `.jsonl`; daily files
/// are `<stem>-YYYYMMDD.jsonl`.
static CFG: OnceLock<(String, u64)> = OnceLock::new();
const MAX_PENDING: usize = 10_000;

fn buffer() -> &'static Mutex<Vec<String>> {
    static B: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    B.get_or_init(|| Mutex::new(Vec::new()))
}

/// Enable the event log (no-op if `path` is empty). Spawns the flusher + pruner.
pub fn init(path: &str, retention_days: u64) {
    let path = path.trim();
    if path.is_empty() {
        return;
    }
    let stem = path.strip_suffix(".jsonl").unwrap_or(path).to_string();
    if CFG.set((stem, retention_days.max(1) * 86_400)).is_err() {
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
    if let Some((stem, r)) = CFG.get() {
        tracing::info!(
            retention_days = r / 86_400,
            "skills/eventlog: enabled — daily files {stem}-YYYYMMDD.jsonl"
        );
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
    let Some((stem, _)) = CFG.get() else {
        return;
    };
    let lines = {
        let mut b = buffer().lock().unwrap();
        std::mem::take(&mut *b)
    };
    if lines.is_empty() {
        return;
    }
    let path = format!("{stem}-{}.jsonl", ymd(now()));
    if let Some(dir) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        for l in lines {
            let _ = writeln!(f, "{l}");
        }
    }
}

/// Delete day-files older than the retention window (by mtime). No rewrite, no race with appends;
/// the current day's file keeps a fresh mtime so it is never pruned.
fn prune() {
    let Some((stem, retention)) = CFG.get() else {
        return;
    };
    let p = std::path::Path::new(stem);
    let Some(base) = p.file_name().and_then(|s| s.to_str()) else {
        return;
    };
    let prefix = format!("{base}-");
    let dir = match p.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => std::path::PathBuf::from("."),
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !(name.starts_with(&prefix) && name.ends_with(".jsonl")) {
            continue;
        }
        let too_old = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| now.duration_since(t).ok())
            .map(|age| age.as_secs() > *retention)
            .unwrap_or(false);
        if too_old && std::fs::remove_file(entry.path()).is_ok() {
            tracing::debug!(file = %name, "skills/eventlog: pruned old day-file");
        }
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn ymd(secs: u64) -> String {
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    format!("{y:04}{m:02}{d:02}")
}

/// Civil date (UTC) from days since the Unix epoch — Howard Hinnant's algorithm (no chrono dep).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (y + if m <= 2 { 1 } else { 0 }, m, d)
}

#[cfg(test)]
mod tests {
    use super::civil_from_days;

    #[test]
    fn civil_from_days_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1)); // epoch
        assert_eq!(civil_from_days(18_993), (2022, 1, 1));
        // 2026-06-20 is 20624 days after the epoch.
        assert_eq!(civil_from_days(20_624), (2026, 6, 20));
    }
}
