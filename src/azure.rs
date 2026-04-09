use anyhow::{bail, Result};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

const COGNITIVE_SERVICES_RESOURCE: &str = "https://cognitiveservices.azure.com";
const TOKEN_REFRESH_MARGIN_SECS: u64 = 300; // Refresh 5 minutes before expiry

#[derive(Debug)]
struct CachedToken {
    access_token: String,
    expires_on: u64,
}

/// Acquires Azure AD tokens via the `az` CLI and caches them.
#[derive(Debug, Clone)]
pub struct AzureCliCredential {
    cached: Arc<RwLock<Option<CachedToken>>>,
}

impl AzureCliCredential {
    pub fn new() -> Self {
        Self {
            cached: Arc::new(RwLock::new(None)),
        }
    }

    /// Returns a valid access token, fetching or refreshing as needed.
    pub async fn get_token(&self) -> Result<String> {
        {
            let cache = self.cached.read().await;
            if let Some(token) = cache.as_ref() {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                if token.expires_on > now + TOKEN_REFRESH_MARGIN_SECS {
                    return Ok(token.access_token.clone());
                }
            }
        }

        let token = Self::fetch_token_from_cli().await?;
        let access_token = token.access_token.clone();

        let mut cache = self.cached.write().await;
        *cache = Some(token);

        Ok(access_token)
    }

    async fn fetch_token_from_cli() -> Result<CachedToken> {
        tracing::debug!("Fetching Azure AD token via az CLI");

        // On Windows, `az` is a .cmd batch file, not a .exe, so we must
        // invoke it through cmd.exe. On Unix, we call `az` directly.
        #[cfg(windows)]
        let output = tokio::process::Command::new("cmd")
            .args([
                "/C",
                "az",
                "account",
                "get-access-token",
                "--resource",
                COGNITIVE_SERVICES_RESOURCE,
                "--output",
                "json",
            ])
            .output()
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to run 'az' CLI via cmd.exe. Is Azure CLI installed and in PATH? Error: {}",
                    e
                )
            })?;

        #[cfg(not(windows))]
        let output = tokio::process::Command::new("az")
            .args([
                "account",
                "get-access-token",
                "--resource",
                COGNITIVE_SERVICES_RESOURCE,
                "--output",
                "json",
            ])
            .output()
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to run 'az' CLI. Is Azure CLI installed and in PATH? Error: {}",
                    e
                )
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "az account get-access-token failed (exit {}): {}. \
                 Make sure you have run 'az login' and have the \
                 'Cognitive Services OpenAI User' role on your Azure OpenAI resource.",
                output.status,
                stderr.trim()
            );
        }

        let json: serde_json::Value = serde_json::from_slice(&output.stdout).map_err(|e| {
            anyhow::anyhow!("Failed to parse az CLI JSON output: {}", e)
        })?;

        let access_token = json["accessToken"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'accessToken' in az CLI output"))?
            .to_string();

        // az CLI returns expires_on as a Unix timestamp (integer or string)
        let expires_on = json["expires_on"]
            .as_u64()
            .or_else(|| json["expires_on"].as_str().and_then(|s| s.parse().ok()))
            .unwrap_or(0);

        tracing::debug!("Azure AD token acquired, expires_on={}", expires_on);

        Ok(CachedToken {
            access_token,
            expires_on,
        })
    }
}

/// Constructs the Azure OpenAI Responses API URL from the endpoint.
///
/// Uses the v1 API: `{endpoint}/openai/v1/responses`
/// The deployment name goes in the request body `model` field.
pub fn azure_responses_url(endpoint: &str) -> String {
    let base = endpoint.trim_end_matches('/');
    format!("{}/openai/v1/responses", base)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn azure_url_construction() {
        assert_eq!(
            azure_responses_url("https://myresource.openai.azure.com"),
            "https://myresource.openai.azure.com/openai/v1/responses"
        );
    }

    #[test]
    fn azure_url_strips_trailing_slash() {
        assert_eq!(
            azure_responses_url("https://myresource.openai.azure.com/"),
            "https://myresource.openai.azure.com/openai/v1/responses"
        );
    }
}
