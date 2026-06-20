mod cli;
mod config;
mod error;
mod metrics;
mod models;
mod proxy;
mod searx;
mod skills;
mod translate;
mod websearch;

use axum::{
    routing::{get, post},
    Extension, Router,
};
use clap::Parser;
use cli::{Cli, Command};
use config::Config;
use daemonize::Daemonize;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto::Builder as ConnBuilder;
use hyper_util::service::TowerToHyperService;
use reqwest::Client;
use std::sync::Arc;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Build version: the CI-injected date-based version (`BUILD_VERSION`) when present,
/// otherwise the crate version from `Cargo.toml`.
pub const VERSION: &str = match option_env!("BUILD_VERSION") {
    Some(v) => v,
    None => env!("CARGO_PKG_VERSION"),
};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if let Some(command) = cli.command {
        match command {
            Command::Stop { pid_file } => {
                stop_daemon(&pid_file)?;
                return Ok(());
            }
            Command::Status { pid_file } => {
                check_status(&pid_file)?;
                return Ok(());
            }
        }
    }

    if cli.daemon {
        use std::fs::OpenOptions;

        let stdout = OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/anthropic-proxy.log")?;

        let stderr = OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/anthropic-proxy.log")?;

        let daemonize = Daemonize::new()
            .pid_file(&cli.pid_file)
            .working_directory(std::env::current_dir()?)
            .stdout(stdout)
            .stderr(stderr)
            .umask(0o027);

        match daemonize.start() {
            Ok(_) => {}
            Err(e) => {
                eprintln!("✗ Failed to daemonize: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("✓ Starting proxy in foreground mode");
    }

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async_main(cli))
}

async fn async_main(cli: Cli) -> anyhow::Result<()> {
    let mut config = Config::from_env_with_path(cli.config)?;

    if cli.debug {
        config.debug = true;
    }
    if cli.verbose {
        config.verbose = true;
    }
    if let Some(port) = cli.port {
        config.port = port;
    }
    if let Some(bind) = cli.bind {
        let trimmed = bind.trim();
        if !trimmed.is_empty() {
            config.bind = trimmed.to_string();
        }
    }
    if !cli.system_prompt_ignore.is_empty() {
        config.system_prompt_ignore_terms.extend(
            cli.system_prompt_ignore
                .into_iter()
                .map(|term| term.trim().to_string())
                .filter(|term| !term.is_empty()),
        );
        Config::dedupe_ignore_terms(&mut config.system_prompt_ignore_terms);
    }

    let log_level = if config.verbose {
        tracing::Level::TRACE
    } else if config.debug {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("anthropic_proxy={}", log_level).into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting Anthropic Proxy v{}", VERSION);
    tracing::info!("Bind: {}", config.bind);
    tracing::info!("Port: {}", config.port);
    tracing::info!("Upstream URLs: {}", config.upstream_urls.join("; "));
    tracing::info!(
        "Resolved chat completions URLs: {}",
        config.chat_completions_urls().join("; ")
    );
    if let Some(ref model) = config.reasoning_model {
        tracing::info!("Reasoning Model Override: {}", model);
    }
    if let Some(ref model) = config.completion_model {
        tracing::info!("Completion Model Override: {}", model);
    }
    if config.passthrough_api_key {
        tracing::info!("API Key: passthrough mode (extracted from x-api-key header)");
    } else if config.api_key.is_some() {
        tracing::info!("API Key: configured");
    } else {
        tracing::info!("API Key: not set (using unauthenticated endpoint)");
    }
    if !config.system_prompt_ignore_terms.is_empty() {
        tracing::info!(
            "System prompt ignore terms: {}",
            config.system_prompt_ignore_terms.join("; ")
        );
    }
    if !config.model_map.is_empty() {
        let entries = config
            .model_map
            .iter()
            .map(|(source, target)| format!("{source} -> {target}"))
            .collect::<Vec<_>>()
            .join("; ");
        tracing::info!("Model map: {}", entries);
    }
    if !config.effort_map.is_empty() {
        tracing::info!("Effort map: thinking budget -> effort level enabled");
    }
    if config.skills.enabled {
        tracing::info!(
            "Skills injection: ENABLED (qdrant={}, collection={}, embed_model={}, top_k={}, tiers=[{}])",
            config.skills.qdrant_url,
            config.skills.collection,
            if config.skills.embed_model.is_empty() {
                "<unset>"
            } else {
                &config.skills.embed_model
            },
            config.skills.top_k,
            config.skills.inject_tiers.join(", "),
        );
    }
    if config.log_requests {
        tracing::info!("Request logging: per-request fields (minus messages/system) enabled");
    }

    let metrics_handle = metrics::install();

    // Keep pooled idle connections short-lived and TCP-keepalive on: a socket the
    // upstream/LB has silently closed is the main cause of intermittent 502s, so we
    // bound how long a stale connection can linger in the pool before reuse.
    // Request timeout aligns with the fronting nginx (proxy_read/send_timeout 600s);
    // a tighter value here would abort long generations the gateway still allows.
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .connect_timeout(std::time::Duration::from_secs(10))
        .pool_max_idle_per_host(10)
        .pool_idle_timeout(std::time::Duration::from_secs(30))
        .tcp_keepalive(std::time::Duration::from_secs(30))
        .build()?;

    let config = Arc::new(config);

    // Compact, persistent learning-event log (no-op unless a path is configured).
    skills::init_eventlog(&config);

    // Stage 3/4/5: background verification, curation, and proactive-learning loops
    // (each a no-op unless skills learning — and, for proactive, ANTHROPIC_PROXY_SKILLS_PROACTIVE — is on).
    skills::spawn_verify(config.clone(), client.clone());
    skills::spawn_curate(config.clone(), client.clone());
    skills::spawn_proactive(config.clone(), client.clone());

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/v1/messages", post(proxy::proxy_handler))
        // Tolerate a trailing slash (some clients append one to ANTHROPIC_BASE_URL).
        .route("/v1/messages/", post(proxy::proxy_handler))
        .route(
            "/v1/messages/count_tokens",
            post(proxy::count_tokens_handler),
        )
        .route("/v1/models", get(proxy::list_models_handler))
        .route("/health", axum::routing::get(health_handler))
        .route(
            "/metrics",
            get(move || {
                let handle = metrics_handle.clone();
                async move { handle.render() }
            }),
        )
        .layer(Extension(config.clone()))
        .layer(Extension(client))
        .layer(TraceLayer::new_for_http())
        .layer(cors);

    let addr = format!("{}:{}", config.bind, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    if config.bind == "0.0.0.0" {
        tracing::warn!(
            "Binding to 0.0.0.0 exposes the proxy on every network interface. \
             The proxy may hold an Anthropic API key, which makes this risky on shared networks. \
             Set --bind 127.0.0.1 (or ANTHROPIC_PROXY_BIND=127.0.0.1) to restrict to localhost."
        );
    }

    tracing::info!("Listening on {}", addr);
    tracing::info!("Proxy ready to accept requests");

    // Manual accept loop instead of `axum::serve`: the latter builds the hyper
    // connection with no timer, which silently disables hyper's built-in 30s
    // header-read timeout (a `Dur::Default` only fires when a `Timer` is installed).
    // Here we install `TokioTimer` and set `header_read_timeout` so a client that
    // connects and then dribbles (or never finishes) its request headers — slowloris —
    // can't pin a connection open indefinitely.
    //
    // The timeout bounds ONLY the request-header read: once headers are in, the handler
    // runs and the streamed response body is unbounded, so long generations (and the
    // heartbeat that keeps them alive) are never cut. This mirrors the Go gateway's
    // `ReadHeaderTimeout: 30s` with the same no-WriteTimeout streaming semantics.
    loop {
        let (tcp, _peer) = match listener.accept().await {
            Ok(conn) => conn,
            Err(err) => {
                // Transient (e.g. EMFILE) — back off briefly so we don't hot-loop.
                tracing::warn!("accept failed: {err}");
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                continue;
            }
        };

        let io = TokioIo::new(tcp);
        let service = TowerToHyperService::new(app.clone());

        tokio::spawn(async move {
            let mut builder = ConnBuilder::new(TokioExecutor::new());
            builder
                .http1()
                .timer(TokioTimer::new()) // required for header_read_timeout to take effect
                .header_read_timeout(std::time::Duration::from_secs(30));

            if let Err(err) = builder.serve_connection_with_upgrades(io, service).await {
                // Usually just a client that opened a socket and sent nothing.
                tracing::trace!("connection error: {err:#}");
            }
        });
    }
}

async fn health_handler() -> &'static str {
    "OK"
}

fn stop_daemon(pid_file: &std::path::Path) -> anyhow::Result<()> {
    if !pid_file.exists() {
        eprintln!("✗ PID file not found: {}", pid_file.display());
        eprintln!("  Daemon is not running or PID file was removed");
        std::process::exit(1);
    }

    let pid_str = std::fs::read_to_string(pid_file)?;
    let pid: i32 = pid_str
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid PID in file: {}", pid_str))?;

    #[cfg(unix)]
    {
        use std::process::Command;
        let output = Command::new("kill").arg(pid.to_string()).output()?;

        if output.status.success() {
            std::fs::remove_file(pid_file)?;
            eprintln!("✓ Daemon stopped (PID: {})", pid);
        } else {
            eprintln!("✗ Failed to stop daemon (PID: {})", pid);
            eprintln!("  Process may have already exited");
            std::fs::remove_file(pid_file)?;
            std::process::exit(1);
        }
    }

    #[cfg(not(unix))]
    {
        eprintln!("✗ Daemon stop is only supported on Unix systems");
        std::process::exit(1);
    }

    Ok(())
}

fn check_status(pid_file: &std::path::Path) -> anyhow::Result<()> {
    if !pid_file.exists() {
        eprintln!("✗ Daemon is not running");
        eprintln!("  PID file not found: {}", pid_file.display());
        std::process::exit(1);
    }

    let pid_str = std::fs::read_to_string(pid_file)?;
    let pid: i32 = pid_str
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid PID in file: {}", pid_str))?;

    #[cfg(unix)]
    {
        use std::process::Command;
        let output = Command::new("ps").arg("-p").arg(pid.to_string()).output()?;

        if output.status.success() {
            eprintln!("✓ Daemon is running (PID: {})", pid);
            eprintln!("  PID file: {}", pid_file.display());
        } else {
            eprintln!("✗ Daemon is not running");
            eprintln!(
                "  Stale PID file found: {} (PID: {})",
                pid_file.display(),
                pid
            );
            std::process::exit(1);
        }
    }

    #[cfg(not(unix))]
    {
        eprintln!("✗ Daemon status check is only supported on Unix systems");
        std::process::exit(1);
    }

    Ok(())
}
