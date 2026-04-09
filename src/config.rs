use anyhow::{bail, Result};
use reqwest::Url;
use std::collections::HashMap;
use std::{env, path::PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model_map: HashMap<String, String>,
    pub debug: bool,
    pub verbose: bool,
    /// Azure OpenAI resource endpoint (e.g. `https://myresource.openai.azure.com`)
    pub azure_openai_endpoint: Option<String>,
    /// Use `az login` CLI credential for Azure OpenAI authentication
    pub azure_use_cli_credential: bool,
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

        if let Ok(exe) = env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                let exe_config = exe_dir.join(".ao-proxy.env");
                if exe_config.exists() && dotenvy::from_path(&exe_config).is_ok() {
                    return Some(exe_config);
                }
            }
        }

        if let Some(home) = Self::home_dir() {
            let home_config = home.join(".ao-proxy.env");
            if home_config.exists() && dotenvy::from_path(&home_config).is_ok() {
                return Some(home_config);
            }
        }

        #[cfg(unix)]
        {
            let etc_config = PathBuf::from("/etc/ao-proxy/.env");
            if etc_config.exists() && dotenvy::from_path(&etc_config).is_ok() {
                return Some(etc_config);
            }
        }

        None
    }

    fn home_dir() -> Option<PathBuf> {
        env::var("HOME")
            .or_else(|_| env::var("USERPROFILE"))
            .ok()
            .map(PathBuf::from)
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
            .unwrap_or(18080);

        let azure_openai_endpoint = env::var("AZURE_OPENAI_ENDPOINT")
            .ok()
            .filter(|v| !v.is_empty());

        let azure_use_cli_credential = env::var("AZURE_USE_CLI_CREDENTIAL")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let base_url = if let Some(ref endpoint) = azure_openai_endpoint {
            // Azure mode: derive base_url from the Azure endpoint
            format!("{}/openai", endpoint.trim_end_matches('/'))
        } else {
            env::var("UPSTREAM_BASE_URL")
                .or_else(|_| env::var("ANTHROPIC_PROXY_BASE_URL"))
                .map_err(|_| {
                    anyhow::anyhow!(
                        "UPSTREAM_BASE_URL (or AZURE_OPENAI_ENDPOINT) is required.\n\
                    Examples:\n\
                      - OpenRouter: https://openrouter.ai/api\n\
                      - OpenAI: https://api.openai.com\n\
                      - Azure: AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com\n\
                      - Local: http://localhost:11434"
                    )
                })?
        };

        Self::validate_base_url(&base_url)?;

        let api_key = env::var("UPSTREAM_API_KEY")
            .or_else(|_| env::var("OPENROUTER_API_KEY"))
            .ok()
            .filter(|k| !k.is_empty());

        let model_map = Self::load_model_map();

        let debug = env::var("DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let verbose = env::var("VERBOSE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        if azure_use_cli_credential && azure_openai_endpoint.is_none() {
            eprintln!(
                "⚠️  AZURE_USE_CLI_CREDENTIAL is set but AZURE_OPENAI_ENDPOINT is not. \
                 CLI credential will be ignored."
            );
        }

        Ok(Config {
            port,
            base_url,
            api_key,
            model_map,
            debug,
            verbose,
            azure_openai_endpoint,
            azure_use_cli_credential,
        })
    }

    fn load_model_map() -> HashMap<String, String> {
        let mut map = Self::default_model_map();

        let raw = match env::var("MODEL_MAP") {
            Ok(val) if !val.is_empty() => val,
            _ => return map,
        };

        if let Ok(loaded) = serde_json::from_str::<HashMap<String, String>>(&raw) {
            map.extend(loaded);
            return map;
        }

        let path = PathBuf::from(&raw);
        match std::fs::read_to_string(&path) {
            Ok(contents) => match serde_json::from_str::<HashMap<String, String>>(&contents) {
                Ok(loaded) => {
                    eprintln!("📄 Loaded model map from: {}", path.display());
                    map.extend(loaded);
                    map
                }
                Err(e) => {
                    eprintln!(
                        "⚠️  Failed to parse model map JSON from {}: {}, using built-in defaults",
                        path.display(),
                        e
                    );
                    map
                }
            },
            Err(_) => {
                eprintln!(
                    "⚠️  MODEL_MAP file not found: {}, using built-in defaults",
                    path.display()
                );
                map
            }
        }
    }

    pub fn responses_url(&self) -> String {
        if let Some(ref endpoint) = self.azure_openai_endpoint {
            return crate::azure::azure_responses_url(endpoint);
        }
        Self::resolve_responses_url(&self.base_url)
            .expect("UPSTREAM_BASE_URL should be validated during configuration loading")
    }

    pub fn is_azure(&self) -> bool {
        self.azure_openai_endpoint.is_some()
    }

    fn validate_base_url(base_url: &str) -> Result<()> {
        Self::resolve_responses_url(base_url).map(|_| ())
    }

    fn resolve_responses_url(base_url: &str) -> Result<String> {
        let (normalized, path_segments) = Self::parse_base_url(base_url)?;

        if Self::is_responses_path(&path_segments) {
            return Ok(normalized.to_string());
        }

        let last_segment = path_segments.last().map(String::as_str);
        if matches!(last_segment, Some("chat") | Some("completions") | Some("response")) {
            bail!(
                "UPSTREAM_BASE_URL must be either a service base URL, a versioned base URL like https://gateway.example.com/v2, or the full .../responses endpoint"
            );
        }

        if last_segment.is_some_and(Self::is_version_segment) {
            return Ok(format!("{}/responses", normalized));
        }

        Ok(format!("{}/v1/responses", normalized))
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

    fn is_responses_path(segments: &[String]) -> bool {
        matches!(segments, [.., responses] if responses == "responses")
    }

    fn is_version_segment(segment: &str) -> bool {
        let version = segment
            .strip_prefix('v')
            .or_else(|| segment.strip_prefix('V'));

        version
            .is_some_and(|value| !value.is_empty() && value.chars().all(|ch| ch.is_ascii_digit()))
    }

    fn default_model_map() -> HashMap<String, String> {
        HashMap::from([
            ("claude-haiku-4-5".to_string(), "gpt-5-mini".to_string()),
            ("claude-sonnet-4-6".to_string(), "gpt-5.4".to_string()),
            ("claude-opus-4-6".to_string(), "gpt-5.4-pro".to_string()),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn base_url_without_version_defaults_to_v1_responses() {
        let url = Config::resolve_responses_url("https://api.openai.com").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/responses");
    }

    #[test]
    fn versioned_base_url_preserves_existing_version() {
        let url = Config::resolve_responses_url("https://gateway.example.com/v2").unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/responses");
    }

    #[test]
    fn full_responses_endpoint_is_used_as_is() {
        let url = Config::resolve_responses_url("https://gateway.example.com/v2/responses/")
            .unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/responses");
    }

    #[test]
    fn chat_completions_endpoint_is_rejected() {
        let err = Config::resolve_responses_url("https://api.openai.com/v1/chat/completions")
            .unwrap_err();
        assert!(err.to_string().contains("full .../responses endpoint"));
    }

    #[test]
    fn query_strings_are_rejected() {
        let err = Config::resolve_responses_url("https://gateway.example.com/v2?foo=bar")
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("must not include query parameters or fragments"));
    }

    #[test]
    fn empty_url_is_rejected() {
        let err = Config::resolve_responses_url("").unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn uppercase_version_segment_is_supported() {
        let url = Config::resolve_responses_url("https://gateway.example.com/V2").unwrap();
        assert_eq!(url, "https://gateway.example.com/V2/responses");
    }

    #[test]
    fn default_model_map_contains_expected_defaults() {
        let map = Config::default_model_map();
        assert_eq!(map.get("claude-haiku-4-5").map(String::as_str), Some("gpt-5-mini"));
        assert_eq!(map.get("claude-sonnet-4-6").map(String::as_str), Some("gpt-5.4"));
        assert_eq!(map.get("claude-opus-4-6").map(String::as_str), Some("gpt-5.4-pro"));
    }

    #[test]
    fn fragments_are_rejected() {
        let err = Config::resolve_responses_url("https://gateway.example.com/v2#section")
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("must not include query parameters or fragments"));
    }

    #[test]
    fn non_http_scheme_is_rejected() {
        let err = Config::resolve_responses_url("ftp://gateway.example.com").unwrap_err();
        assert!(err.to_string().contains("must use http or https"));
    }

    #[test]
    fn explicit_v1_is_preserved_not_doubled() {
        let url = Config::resolve_responses_url("https://openrouter.ai/api/v1").unwrap();
        assert_eq!(url, "https://openrouter.ai/api/v1/responses");
    }

    #[test]
    fn trailing_slash_on_base_url_is_normalized() {
        let url = Config::resolve_responses_url("https://api.openai.com/").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/responses");
    }

    #[test]
    fn url_with_subpath_and_no_version_defaults_to_v1() {
        let url = Config::resolve_responses_url("https://openrouter.ai/api").unwrap();
        assert_eq!(url, "https://openrouter.ai/api/v1/responses");
    }

    #[test]
    fn only_completions_path_is_rejected() {
        let err = Config::resolve_responses_url("https://gateway.example.com/v2/completions")
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("service base URL, a versioned base URL"));
    }

    #[test]
    fn uppercase_version_prefix_is_accepted() {
        let url = Config::resolve_responses_url("https://gateway.example.com/V2").unwrap();
        assert_eq!(url, "https://gateway.example.com/V2/responses");
    }
}
