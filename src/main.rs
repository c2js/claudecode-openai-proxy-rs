mod azure;
mod cli;
mod config;
mod error;
mod models;
mod proxy;
mod transform;

use axum::{
    routing::{get, post},
    Extension, Router,
};
use clap::Parser;
use cli::Cli;
use config::Config;
use reqwest::Client;
use std::sync::Arc;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use tracing_subscriber::fmt::time::FormatTime;

struct SecondTimer;

struct ShortTargetFormatter;

impl<S, N> tracing_subscriber::fmt::FormatEvent<S, N> for ShortTargetFormatter
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'a> tracing_subscriber::fmt::FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        mut writer: tracing_subscriber::fmt::format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        use tracing::Level;

        // Dimmed timestamp
        write!(writer, "\x1b[2m")?;
        SecondTimer.format_time(&mut writer)?;
        write!(writer, "\x1b[0m")?;

        let level = *event.metadata().level();
        let color = match level {
            Level::ERROR => "\x1b[31m", // red
            Level::WARN  => "\x1b[33m", // yellow
            Level::INFO  => "\x1b[32m", // green
            Level::DEBUG => "\x1b[34m", // blue
            Level::TRACE => "\x1b[35m", // magenta
        };
        write!(writer, " {}{:>5}\x1b[0m ", color, level)?;

        let target = event.metadata().target();
        let short_target = target.rsplit("::").next().unwrap_or(target);
        write!(writer, "\x1b[2m{}:\x1b[0m ", short_target)?;
        _ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

impl tracing_subscriber::fmt::time::FormatTime for SecondTimer {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = now.as_secs();
        // 2026-04-09T14:00:59Z
        let days = secs / 86400;
        let time_secs = secs % 86400;
        let h = time_secs / 3600;
        let m = (time_secs % 3600) / 60;
        let s = time_secs % 60;
        // Days since epoch to Y-M-D
        let (y, mo, d) = epoch_days_to_ymd(days as i64);
        write!(w, "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, m, s)
    }
}

fn epoch_days_to_ymd(mut days: i64) -> (i64, u32, u32) {
    // Civil days algorithm from Howard Hinnant
    days += 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let doe = (days - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("✓ Starting proxy in foreground mode");

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
                .unwrap_or_else(|_| format!("ao_proxy={}", log_level).into()),
        )
        .with(tracing_subscriber::fmt::layer()
            .with_timer(SecondTimer)
            .with_target(true)
            .fmt_fields(tracing_subscriber::fmt::format::DefaultFields::new())
            .event_format(ShortTargetFormatter)
        )
        .init();

    tracing::info!("Starting Anthropic Proxy v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Port: {}", config.port);
    tracing::info!("Upstream URL: {}", config.base_url);
    tracing::info!(
        "Resolved upstream Responses URL: {}",
        config.responses_url()
    );
    if !config.model_map.is_empty() {
        tracing::info!("Model map: {} entries loaded", config.model_map.len());
        for (from, to) in &config.model_map {
            tracing::info!("  {} -> {}", from, to);
        }
    }
    if config.is_azure() {
        tracing::info!(
            "Azure OpenAI: endpoint={}",
            config.azure_openai_endpoint.as_deref().unwrap_or("?")
        );
        if config.azure_use_cli_credential {
            tracing::info!("Azure auth: az CLI credential (az login)");
        } else if config.api_key.is_some() {
            tracing::info!("Azure auth: API key");
        } else {
            tracing::info!("Azure auth: none configured");
        }
    } else if config.api_key.is_some() {
        tracing::info!("API Key: configured");
    } else {
        tracing::info!("API Key: not set (using unauthenticated endpoint)");
    }

    // Create Azure CLI credential if configured
    let azure_credential = if config.azure_use_cli_credential && config.is_azure() {
        let cred = azure::AzureCliCredential::new();
        // Validate credential on startup
        match cred.get_token().await {
            Ok(_) => tracing::info!("Azure CLI credential: validated"),
            Err(e) => tracing::warn!("Azure CLI credential: initial token fetch failed ({}). Will retry on first request.", e),
        }
        Some(cred)
    } else {
        None
    };

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .connect_timeout(std::time::Duration::from_secs(10))
        .pool_max_idle_per_host(10)
        .build()?;

    let config = Arc::new(config);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/v1/messages", post(proxy::proxy_handler))
        .route("/v1/models", get(proxy::list_models_handler))
        .route("/health", axum::routing::get(health_handler))
        .layer(Extension(config.clone()))
        .layer(Extension(client))
        .layer(Extension(azure_credential))
        .layer(TraceLayer::new_for_http())
        .layer(cors);

    let addr = format!("0.0.0.0:{}", config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!("Listening on {}", addr);
    tracing::info!("Proxy ready to accept requests");

    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_handler() -> &'static str {
    "OK"
}
