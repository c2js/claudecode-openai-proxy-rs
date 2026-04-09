# claudecode-openai-proxy-rs

Rust proxy that accepts Anthropic Messages API requests and forwards them to an OpenAI-compatible Responses API upstream.

Derived from the original `anthropic-proxy-rs` project by m0n0x41d.

The proxy is intentionally narrow:

- Anthropic-facing models: `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-6`
- Upstream-facing models: any `gpt-5*` family string such as `gpt-5`, `gpt-5-mini`, `gpt-5.4`, `gpt-5.4-pro`, or `openai/gpt-5.4`
- Upstream generation endpoint: `POST /v1/responses`

This is aimed at Claude Code and other Anthropic-compatible clients that need to talk to GPT-5 family backends through an Anthropic-compatible surface.

## Features

- Anthropic-compatible `POST /v1/messages`
- Anthropic-compatible `GET /v1/models`
- OpenAI Responses API upstream integration
- Streaming SSE translation from Responses events to Anthropic message events
- Tool call and tool result translation
- Thinking and reasoning effort translation
- Structured output passthrough via `output_config.format` -> `text.format`
- Static Claude model catalog limited to the supported models

## Supported Models

Anthropic request models:

- `claude-haiku-4-5`
- `claude-sonnet-4-6`
- `claude-opus-4-6`

Accepted aliases:

- Date suffixes are stripped during lookup, so snapshot-style values such as `claude-haiku-4-5-20251001` are also accepted.

Upstream model rules:

- The final path segment must start with `gpt-5`
- Examples: `gpt-5`, `gpt-5-mini`, `gpt-5.4`, `gpt-5.4-pro`, `openai/gpt-5.4`

## Default Model Map

If `MODEL_MAP` is not set, the proxy uses these built-in defaults:

```json
{
  "claude-haiku-4-5": "gpt-5-mini",
  "claude-sonnet-4-6": "gpt-5.4",
  "claude-opus-4-6": "gpt-5.4-pro"
}
```

If `MODEL_MAP` is set, it is merged over those defaults.

## Quick Start

### Linux / macOS

Build and run manually:

```bash
cargo build --release
cp .env.example .env
./target/release/ao-proxy
```

With Claude Code:

```bash
# Terminal 1
ao-proxy

# Terminal 2
ANTHROPIC_BASE_URL=http://localhost:18080 ANTHROPIC_API_KEY="any-value" claude
```

> **Note:** Claude Code requires `ANTHROPIC_API_KEY` to be set. Since authentication is handled by the upstream provider (via `UPSTREAM_API_KEY`), this proxy does not validate the key — any non-empty value will work (e.g. `ANTHROPIC_API_KEY="any-value"`).

### Windows

Prerequisites:
- **Rust (1.70+)** — Download from https://rustup.rs/
- **Visual Studio C++ Build Tools** — Required for the MSVC linker and Windows SDK. Install via [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select the "Desktop development with C++" workload. This provides `link.exe` and the Windows system libraries that Rust needs.
- **Git for Windows**

Alternatively, if you prefer the GNU toolchain instead of MSVC:

```bash
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
```

This uses MinGW's `gcc` linker and avoids needing Visual Studio, but MSVC is the default and recommended target on Windows.

Build:

```powershell
cargo build --release
Copy-Item .env.example .env
.\target\release\ao-proxy.exe
```

With Claude Code on Windows (PowerShell):

```powershell
# Terminal 1
.\target\release\ao-proxy.exe

# Terminal 2
$env:ANTHROPIC_BASE_URL='http://localhost:18080'
$env:ANTHROPIC_API_KEY='any-value'
claude
```

With Claude Code on Windows (Git Bash / WSL):

```bash
# Terminal 1
./target/release/ao-proxy.exe

# Terminal 2
ANTHROPIC_BASE_URL=http://localhost:18080 ANTHROPIC_API_KEY="any-value" claude
```

## Configuration

### Command Line Options

```bash
ao-proxy --help
```

Commands:

*None — the proxy runs in foreground mode only.*

Options:

| Option | Short | Description |
|--------|-------|-------------|
| `--config <FILE>` | `-c` | Path to custom `.env` file |
| `--debug` | `-d` | Enable debug logging |
| `--verbose` | `-v` | Enable verbose request and response logging |
| `--port <PORT>` | `-p` | Port to listen on |

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `UPSTREAM_BASE_URL` | Yes* | - | OpenAI-compatible base URL or full Responses endpoint |
| `UPSTREAM_API_KEY` | No | - | Bearer token for upstream service (or Azure API key) |
| `MODEL_MAP` | No | Built-in defaults | Inline JSON or path to JSON file |
| `PORT` | No | `18080` | Listen port |
| `DEBUG` | No | `false` | Enable debug logs |
| `VERBOSE` | No | `false` | Enable verbose logs |
| `AZURE_OPENAI_ENDPOINT` | No | - | Azure OpenAI resource endpoint (replaces `UPSTREAM_BASE_URL`) |
| `AZURE_USE_CLI_CREDENTIAL` | No | `false` | Use `az login` for Azure authentication |

\* Not required when `AZURE_OPENAI_ENDPOINT` is set.

`UPSTREAM_BASE_URL` accepts:

- `https://api.openai.com` -> `https://api.openai.com/v1/responses`
- `https://openrouter.ai/api` -> `https://openrouter.ai/api/v1/responses`
- `https://gateway.example.com/v2` -> `https://gateway.example.com/v2/responses`
- `https://gateway.example.com/v2/responses` -> used as-is

The proxy searches for `.env` files in this order:

1. Path from `--config`
2. Current working directory as `.env`
3. `~/.ao-proxy.env` (`%USERPROFILE%\.ao-proxy.env` on Windows)
4. `/etc/ao-proxy/.env` (Unix only)

## Azure OpenAI

The proxy supports Azure OpenAI with either API key or Azure CLI credential (`az login`) authentication.

### Setup

1. Set `AZURE_OPENAI_ENDPOINT` to your Azure OpenAI resource endpoint.
2. Set `MODEL_MAP` values to your Azure deployment names.
3. Choose an authentication method (see below).

The proxy uses the Azure OpenAI v1 API: `POST {endpoint}/openai/v1/responses`.

### Authentication

**Option 1: Azure CLI credential (recommended for local development)**

```bash
# Prerequisites: Azure CLI installed, logged in, and RBAC role assigned
az login
az role assignment create \
  --assignee <your-user-or-principal-id> \
  --role "Cognitive Services OpenAI User" \
  --scope /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<resource>
```

```bash
AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com \
AZURE_USE_CLI_CREDENTIAL=true \
MODEL_MAP='{"claude-sonnet-4-6":"my-gpt5-deployment"}' \
ao-proxy
```

The proxy automatically acquires and caches Azure AD tokens, refreshing them before expiry.

**Option 2: API key**

```bash
AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com \
UPSTREAM_API_KEY=<your-azure-api-key> \
MODEL_MAP='{"claude-sonnet-4-6":"my-gpt5-deployment"}' \
ao-proxy
```

When using Azure with an API key, the proxy sends it via the `api-key` header (Azure's convention) instead of `Authorization: Bearer`.

### Azure with Claude Code

```bash
# Terminal 1
AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com \
AZURE_USE_CLI_CREDENTIAL=true \
MODEL_MAP='{"claude-sonnet-4-6":"my-gpt5-deployment"}' \
ao-proxy

# Terminal 2
ANTHROPIC_BASE_URL=http://localhost:18080 ANTHROPIC_API_KEY="any-value" claude
```

## Example `.env`

See [.env.example](.env.example).

Inline override example:

```bash
MODEL_MAP='{"claude-haiku-4-5":"gpt-5-mini","claude-sonnet-4-6":"gpt-5.4","claude-opus-4-6":"gpt-5.4-pro"}' \
UPSTREAM_BASE_URL=https://openrouter.ai/api \
UPSTREAM_API_KEY=sk-or-... \
ao-proxy
```

File-based override example:

```bash
MODEL_MAP=/etc/ao-proxy/model-map.json ao-proxy
```

## Mapping Behavior

Request mapping:

- Anthropic `messages` -> Responses `input`
- Anthropic `system` -> Responses `message` items with `role: "system"`
- Anthropic `tool_use` history -> Responses `function_call` items
- Anthropic `tool_result` history -> Responses `function_call_output` items
- Anthropic `thinking` -> Responses `reasoning` config and assistant `commentary` history items
- Anthropic `output_config.effort` -> Responses `reasoning.effort`
- Anthropic `output_config.format` -> Responses `text.format`
- Anthropic `tool_choice` -> Responses `tool_choice`
- Anthropic `service_tier` -> Responses `service_tier`
- Anthropic `metadata` -> stringified Responses metadata pairs

Reasoning effort mapping:

| Anthropic | Responses |
|-----------|-----------|
| `low` | `low` (`medium` for `gpt-5.4-pro*`) |
| `medium` | `medium` |
| `high` | `high` |
| `max` | `xhigh` |

Response mapping:

- Responses `message.output_text` -> Anthropic `text` content blocks
- Responses `reasoning` summaries -> Anthropic `thinking` content blocks
- Responses `function_call` items -> Anthropic `tool_use` content blocks

## Streaming

The proxy translates Responses streaming events into Anthropic message stream events.

Handled upstream event families include:

- `response.output_text.delta`
- `response.reasoning_summary_text.delta`
- `response.reasoning_text.delta`
- `response.function_call_arguments.delta`
- `response.completed`
- `response.incomplete`
- `response.failed`

## Known Incompatibilities

These are the important gaps that remain after the revamp:

- Anthropic `top_k` is not supported by the Responses API. Requests using it are rejected.
- Anthropic `stop_sequences` are not supported by the Responses API. Requests using them are rejected.
- Anthropic `max_tokens` values below 16 are clamped to 16 to meet the OpenAI `max_output_tokens` minimum. This commonly occurs with Claude Code's `/context` command, which sends `max_tokens: 1`.
- GPT-5 family sampling with reasoning is restricted upstream. When thinking is enabled, `temperature` and `top_p` are dropped to avoid upstream errors.
- Anthropic `tool_result.is_error` has no native Responses equivalent. The proxy encodes the error into the function call output payload.
- Anthropic thinking history is not lossless when replayed upstream. It is approximated as assistant commentary text.
- Anthropic client tools map to Responses custom function tools. Anthropic server-side tools and special tool types are not translated.
- Responses-native built-in tools such as web search, file search, MCP, code interpreter, and image generation are not exposed through the Anthropic-facing API surface.
- Responses citations, refusal-specific output structure, and pause-turn semantics are not fully represented in the Anthropic response shape.
- The proxy returns only the three Claude-facing models from `/v1/models`; it does not query upstream model availability.
- Responses are sent with `store: false` to keep the proxy stateless and closer to Anthropic-style manual history replay.

## Troubleshooting

Wrong upstream path:

- `https://api.openai.com` resolves to `/v1/responses`
- `https://openrouter.ai/api` resolves to `/v1/responses`
- Full `/responses` endpoints are used as-is
- Partial paths and URLs with query strings or fragments are rejected

Model rejected:

- Anthropic-side requests must be one of the three supported Claude models
- Upstream mapped model strings must be in the `gpt-5*` family

Tool or streaming issues:

- Verify the upstream provider supports the Responses API, function calling, and streaming for the mapped model

## Development

```bash
cargo test
```

## License

MIT. See [LICENSE](LICENSE).
