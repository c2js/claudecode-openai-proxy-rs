use crate::azure::AzureCliCredential;
use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::models::{anthropic, responses};
use crate::transform;
use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue},
    response::{IntoResponse, Response},
    Extension, Json,
};
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

pub async fn proxy_handler(
    Extension(config): Extension<Arc<Config>>,
    Extension(client): Extension<Client>,
    Extension(azure_cred): Extension<Option<AzureCliCredential>>,
    Json(req): Json<anthropic::AnthropicRequest>,
) -> ProxyResult<Response> {
    let is_streaming = req.stream.unwrap_or(false);

    let anthropic_effort = req
        .output_config
        .as_ref()
        .and_then(|c| c.effort.as_deref())
        .or_else(|| {
            req.thinking.as_ref().and_then(|t| match t.thinking_type.as_str() {
                "enabled" | "adaptive" => match t.budget_tokens {
                    Some(0..=2048) => Some("low (from budget_tokens)"),
                    Some(2049..=8192) => Some("medium (from budget_tokens)"),
                    Some(8193..=32768) => Some("high (from budget_tokens)"),
                    Some(_) => Some("xhigh (from budget_tokens)"),
                    None => Some("high (default)"),
                },
                _ => None,
            })
        })
        .unwrap_or("none")
        .to_string();
    let anthropic_model = req.model.clone();

    tracing::debug!("Received request for model: {}", req.model);
    tracing::debug!("Streaming: {}", is_streaming);

    if let Some(ref thinking) = req.thinking {
        tracing::debug!(
            "Anthropic thinking: type={}, budget_tokens={:?}",
            thinking.thinking_type,
            thinking.budget_tokens
        );
    }
    if let Some(ref output) = req.output_config {
        if let Some(ref effort) = output.effort {
            tracing::debug!("Anthropic reasoning effort: {}", effort);
        }
    }

    if config.verbose {
        tracing::trace!(
            "Incoming Anthropic request: {}",
            serde_json::to_string_pretty(&req).unwrap_or_default()
        );
    }

    let prepared = transform::anthropic_to_responses(req, &config)?;

    let openai_effort = prepared
        .upstream_request
        .reasoning
        .as_ref()
        .map(|r| r.effort.as_str())
        .unwrap_or("none");

    tracing::info!(
        "{} [effort: {}] -> {} [effort: {}]",
        anthropic_model,
        anthropic_effort,
        prepared.upstream_model,
        openai_effort,
    );

    if let Some(ref reasoning) = prepared.upstream_request.reasoning {
        tracing::debug!(
            "OpenAI reasoning effort: {} (model: {})",
            reasoning.effort,
            prepared.upstream_model
        );
    }

    if config.verbose {
        tracing::trace!(
            "Transformed Responses request: {}",
            serde_json::to_string_pretty(&prepared.upstream_request).unwrap_or_default()
        );
    }

    if is_streaming {
        handle_streaming(config, client, azure_cred, prepared).await
    } else {
        handle_non_streaming(config, client, azure_cred, prepared).await
    }
}

pub async fn list_models_handler() -> ProxyResult<Response> {
    Ok(Json(transform::supported_models_response()).into_response())
}

/// Applies the appropriate authentication header to the request.
async fn apply_auth(
    config: &Config,
    azure_cred: &Option<AzureCliCredential>,
    req_builder: reqwest::RequestBuilder,
) -> ProxyResult<reqwest::RequestBuilder> {
    // Azure CLI credential takes priority when configured
    if config.is_azure() && config.azure_use_cli_credential {
        if let Some(cred) = azure_cred {
            let token = cred.get_token().await.map_err(|e| {
                ProxyError::Config(format!("Azure CLI credential error: {}", e))
            })?;
            return Ok(req_builder.header("Authorization", format!("Bearer {}", token)));
        }
    }

    if let Some(api_key) = &config.api_key {
        if config.is_azure() {
            // Azure OpenAI uses the api-key header for key-based auth
            return Ok(req_builder.header("api-key", api_key.as_str()));
        }
        // Standard OpenAI-compatible: Bearer token
        return Ok(req_builder.header("Authorization", format!("Bearer {}", api_key)));
    }

    Ok(req_builder)
}

async fn handle_non_streaming(
    config: Arc<Config>,
    client: Client,
    azure_cred: Option<AzureCliCredential>,
    prepared: transform::PreparedRequest,
) -> ProxyResult<Response> {
    let url = config.responses_url();
    tracing::debug!("Sending non-streaming request to {}", url);
    tracing::debug!("Request model: {}", prepared.upstream_model);

    let mut req_builder = client
        .post(&url)
        .json(&prepared.upstream_request)
        .timeout(Duration::from_secs(300));

    req_builder = apply_auth(&config, &azure_cred, req_builder).await?;

    let response = req_builder.send().await.map_err(|err| {
        tracing::error!("Failed to send non-streaming request to {}: {:?}", url, err);
        ProxyError::Http(err)
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        tracing::error!("Upstream error ({}): {}", status, error_text);
        return Err(ProxyError::Upstream(format!(
            "Upstream returned {}: {}",
            status, error_text
        )));
    }

    let upstream_resp: responses::ResponsesResponse = response.json().await?;

    if config.verbose {
        tracing::trace!(
            "Received Responses response: {}",
            serde_json::to_string_pretty(&upstream_resp).unwrap_or_default()
        );
    }

    let openai_model = upstream_resp.model.as_deref().unwrap_or("unknown");
    let usage = upstream_resp.usage.as_ref();
    let input_tokens = usage.map(|u| u.input_tokens).unwrap_or(0);
    let output_tokens = usage.map(|u| u.output_tokens).unwrap_or(0);
    let reasoning_tokens = usage
        .and_then(|u| u.output_tokens_details.as_ref())
        .and_then(|d| d.reasoning_tokens)
        .unwrap_or(0);
    tracing::info!(
        "{} <- {} id={} tokens(i={}, o={}, r={})",
        prepared.anthropic_model,
        openai_model,
        truncate_resp_id(&upstream_resp.id),
        input_tokens,
        output_tokens,
        reasoning_tokens,
    );

    let anthropic_resp =
        transform::responses_to_anthropic(upstream_resp, &prepared.anthropic_model)?;

    if config.verbose {
        tracing::trace!(
            "Transformed Anthropic response: {}",
            serde_json::to_string_pretty(&anthropic_resp).unwrap_or_default()
        );
    }

    Ok(Json(anthropic_resp).into_response())
}

async fn handle_streaming(
    config: Arc<Config>,
    client: Client,
    azure_cred: Option<AzureCliCredential>,
    prepared: transform::PreparedRequest,
) -> ProxyResult<Response> {
    let url = config.responses_url();
    tracing::debug!("Sending streaming request to {}", url);
    tracing::debug!("Request model: {}", prepared.upstream_model);

    let mut req_builder = client
        .post(&url)
        .json(&prepared.upstream_request)
        .timeout(Duration::from_secs(300));

    req_builder = apply_auth(&config, &azure_cred, req_builder).await?;

    let response = req_builder.send().await.map_err(|err| {
        tracing::error!("Failed to send streaming request to {}: {:?}", url, err);
        ProxyError::Http(err)
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        tracing::error!("Upstream error ({}) from {}: {}", status, url, error_text);
        return Err(ProxyError::Upstream(format!(
            "Upstream returned {} from {}: {}",
            status, url, error_text
        )));
    }

    let stream = response.bytes_stream();
    let sse_stream = create_sse_stream(stream, prepared.anthropic_model);

    let mut headers = HeaderMap::new();
    headers.insert(
        "Content-Type",
        HeaderValue::from_static("text/event-stream"),
    );
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
    headers.insert("Connection", HeaderValue::from_static("keep-alive"));

    Ok((headers, Body::from_stream(sse_stream)).into_response())
}

fn create_sse_stream(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    anthropic_model: String,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut response_id: Option<String> = None;
        let mut message_started = false;
        let mut content_index = 0usize;
        let mut current_block: Option<BlockKind> = None;
        let mut function_calls: HashMap<String, FunctionCallState> = HashMap::new();

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    buffer.push_str(&text);
                    buffer = buffer.replace("\r\n", "\n");

                    while let Some(block) = take_sse_block(&mut buffer) {
                        let data = extract_sse_data(&block);
                        if data.trim().is_empty() {
                            continue;
                        }

                        let event = match serde_json::from_str::<responses::ResponseStreamEvent>(&data) {
                            Ok(event) => event,
                            Err(_) => {
                                tracing::debug!("Ignoring unrecognized upstream stream event: {}", data);
                                continue;
                            }
                        };

                        match event {
                            responses::ResponseStreamEvent::ResponseCreated { response } => {
                                response_id = Some(response.id);
                            }
                            responses::ResponseStreamEvent::ResponseInProgress { response } => {
                                response_id = Some(response.id);
                            }
                            responses::ResponseStreamEvent::OutputItemAdded { item, .. } => {
                                if let Some(start_event) = ensure_message_start(
                                    &mut message_started,
                                    response_id.as_deref(),
                                    &anthropic_model,
                                ) {
                                    yield Ok(Bytes::from(start_event));
                                }

                                if let responses::StreamOutputItem::FunctionCall { id, call_id, name, .. } = item {
                                    for event in switch_to_tool_use(
                                        &mut current_block,
                                        &mut content_index,
                                        id.clone(),
                                        &call_id,
                                        &name,
                                    ) {
                                        yield Ok(Bytes::from(event));
                                    }

                                    function_calls.insert(
                                        id,
                                        FunctionCallState {
                                            call_id,
                                            name,
                                            saw_arguments: false,
                                            emitted: true,
                                        },
                                    );
                                }
                            }
                            responses::ResponseStreamEvent::OutputItemDone { item, .. } => {
                                if let responses::StreamOutputItem::FunctionCall { id, call_id, name, arguments, .. } = item {
                                    let state = function_calls.entry(id.clone()).or_insert(FunctionCallState {
                                        call_id,
                                        name,
                                        saw_arguments: false,
                                        emitted: false,
                                    });

                                    if !state.emitted {
                                        for event in switch_to_tool_use(
                                            &mut current_block,
                                            &mut content_index,
                                            id.clone(),
                                            &state.call_id,
                                            &state.name,
                                        ) {
                                            yield Ok(Bytes::from(event));
                                        }
                                        state.emitted = true;
                                    }

                                    if let Some(arguments) = arguments {
                                        if !state.saw_arguments {
                                            yield Ok(Bytes::from(render_event(
                                                "content_block_delta",
                                                json!({
                                                    "type": "content_block_delta",
                                                    "index": content_index,
                                                    "delta": {
                                                        "type": "input_json_delta",
                                                        "partial_json": arguments,
                                                    }
                                                }),
                                            )));
                                            state.saw_arguments = true;
                                        }
                                    }

                                    if let Some(stop) = close_current_block(&mut current_block, &mut content_index) {
                                        yield Ok(Bytes::from(stop));
                                    }
                                }
                            }
                            responses::ResponseStreamEvent::OutputTextDelta { delta, .. } => {
                                if let Some(start_event) = ensure_message_start(
                                    &mut message_started,
                                    response_id.as_deref(),
                                    &anthropic_model,
                                ) {
                                    yield Ok(Bytes::from(start_event));
                                }

                                for event in switch_to_text(&mut current_block, &mut content_index) {
                                    yield Ok(Bytes::from(event));
                                }

                                if !delta.is_empty() {
                                    yield Ok(Bytes::from(render_event(
                                        "content_block_delta",
                                        json!({
                                            "type": "content_block_delta",
                                            "index": content_index,
                                            "delta": {
                                                "type": "text_delta",
                                                "text": delta,
                                            }
                                        }),
                                    )));
                                }
                            }
                            responses::ResponseStreamEvent::OutputTextDone { text, .. } => {
                                if !text.is_empty() {
                                    for event in switch_to_text(&mut current_block, &mut content_index) {
                                        yield Ok(Bytes::from(event));
                                    }
                                }

                                if let Some(stop) = close_current_block(&mut current_block, &mut content_index) {
                                    yield Ok(Bytes::from(stop));
                                }
                            }
                            responses::ResponseStreamEvent::ContentPartAdded { .. }
                            | responses::ResponseStreamEvent::ContentPartDone { .. }
                            | responses::ResponseStreamEvent::ReasoningSummaryPartAdded { .. }
                            | responses::ResponseStreamEvent::ReasoningSummaryPartDone { .. }
                            | responses::ResponseStreamEvent::Keepalive { .. } => {}
                            responses::ResponseStreamEvent::ReasoningSummaryTextDelta { delta, .. }
                            | responses::ResponseStreamEvent::ReasoningTextDelta { delta, .. } => {
                                if let Some(start_event) = ensure_message_start(
                                    &mut message_started,
                                    response_id.as_deref(),
                                    &anthropic_model,
                                ) {
                                    yield Ok(Bytes::from(start_event));
                                }

                                for event in switch_to_thinking(&mut current_block, &mut content_index) {
                                    yield Ok(Bytes::from(event));
                                }

                                if !delta.is_empty() {
                                    yield Ok(Bytes::from(render_event(
                                        "content_block_delta",
                                        json!({
                                            "type": "content_block_delta",
                                            "index": content_index,
                                            "delta": {
                                                "type": "thinking_delta",
                                                "thinking": delta,
                                            }
                                        }),
                                    )));
                                }
                            }
                            responses::ResponseStreamEvent::ReasoningSummaryTextDone { text, .. }
                            | responses::ResponseStreamEvent::ReasoningTextDone { text, .. } => {
                                if !text.is_empty() {
                                    for event in switch_to_thinking(&mut current_block, &mut content_index) {
                                        yield Ok(Bytes::from(event));
                                    }
                                }

                                if let Some(stop) = close_current_block(&mut current_block, &mut content_index) {
                                    yield Ok(Bytes::from(stop));
                                }
                            }
                            responses::ResponseStreamEvent::FunctionCallArgumentsDelta {
                                item_id,
                                delta,
                                ..
                            } => {
                                if let Some(start_event) = ensure_message_start(
                                    &mut message_started,
                                    response_id.as_deref(),
                                    &anthropic_model,
                                ) {
                                    yield Ok(Bytes::from(start_event));
                                }

                                if let Some(state) = function_calls.get_mut(&item_id) {
                                    for event in switch_to_tool_use(
                                        &mut current_block,
                                        &mut content_index,
                                        item_id.clone(),
                                        &state.call_id,
                                        &state.name,
                                    ) {
                                        yield Ok(Bytes::from(event));
                                    }

                                    if !delta.is_empty() {
                                        yield Ok(Bytes::from(render_event(
                                            "content_block_delta",
                                            json!({
                                                "type": "content_block_delta",
                                                "index": content_index,
                                                "delta": {
                                                    "type": "input_json_delta",
                                                    "partial_json": delta,
                                                }
                                            }),
                                        )));
                                        state.saw_arguments = true;
                                        state.emitted = true;
                                    }
                                }
                            }
                            responses::ResponseStreamEvent::FunctionCallArgumentsDone {
                                item_id,
                                name,
                                arguments,
                                ..
                            } => {
                                let state = function_calls.entry(item_id.clone()).or_insert(FunctionCallState {
                                    call_id: item_id.clone(),
                                    name,
                                    saw_arguments: false,
                                    emitted: false,
                                });

                                for event in switch_to_tool_use(
                                    &mut current_block,
                                    &mut content_index,
                                    item_id.clone(),
                                    &state.call_id,
                                    &state.name,
                                ) {
                                    yield Ok(Bytes::from(event));
                                }

                                if !state.saw_arguments && !arguments.is_empty() {
                                    yield Ok(Bytes::from(render_event(
                                        "content_block_delta",
                                        json!({
                                            "type": "content_block_delta",
                                            "index": content_index,
                                            "delta": {
                                                "type": "input_json_delta",
                                                "partial_json": arguments,
                                            }
                                        }),
                                    )));
                                    state.saw_arguments = true;
                                }

                                state.emitted = true;

                                if let Some(stop) = close_current_block(&mut current_block, &mut content_index) {
                                    yield Ok(Bytes::from(stop));
                                }
                            }
                            responses::ResponseStreamEvent::ResponseCompleted { response }
                            | responses::ResponseStreamEvent::ResponseIncomplete { response } => {
                                let openai_model = response.model.as_deref().unwrap_or("unknown");
                                let usage = response.usage.as_ref();
                                let input_tokens = usage.map(|u| u.input_tokens).unwrap_or(0);
                                let output_tokens = usage.map(|u| u.output_tokens).unwrap_or(0);
                                let reasoning_tokens = usage
                                    .and_then(|u| u.output_tokens_details.as_ref())
                                    .and_then(|d| d.reasoning_tokens)
                                    .unwrap_or(0);
                                tracing::info!(
                                    "{} <- {} id={} tokens(i={}, o={}, r={})",
                                    anthropic_model,
                                    openai_model,
                                    truncate_resp_id(&response.id),
                                    input_tokens,
                                    output_tokens,
                                    reasoning_tokens,
                                );

                                if let Some(stop) = close_current_block(&mut current_block, &mut content_index) {
                                    yield Ok(Bytes::from(stop));
                                }

                                let saw_tool_use = function_calls.values().any(|state| state.emitted);
                                yield Ok(Bytes::from(message_delta_event(
                                    transform::map_responses_stop_reason(
                                        response.status.as_deref(),
                                        response
                                            .incomplete_details
                                            .as_ref()
                                            .map(|details| details.reason.as_str()),
                                        saw_tool_use,
                                    ),
                                    response.usage.as_ref(),
                                )));
                                yield Ok(Bytes::from(render_event(
                                    "message_stop",
                                    json!({ "type": "message_stop" }),
                                )));
                                break;
                            }
                            responses::ResponseStreamEvent::ResponseFailed { response } => {
                                if let Some(stop) = close_current_block(&mut current_block, &mut content_index) {
                                    yield Ok(Bytes::from(stop));
                                }

                                let message = response
                                    .error
                                    .as_ref()
                                    .map(|error| error.message.clone())
                                    .unwrap_or_else(|| "Upstream response failed".to_string());
                                yield Ok(Bytes::from(error_event("upstream_error", &message)));
                                break;
                            }
                            responses::ResponseStreamEvent::Error { code, message, .. } => {
                                yield Ok(Bytes::from(error_event(&code, &message)));
                                break;
                            }
                        }
                    }
                }
                Err(err) => {
                    tracing::error!("Stream error: {}", err);
                    yield Ok(Bytes::from(error_event(
                        "stream_error",
                        &format!("Stream error: {}", err),
                    )));
                    break;
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
enum BlockKind {
    Text,
    Thinking,
    ToolUse { item_id: String },
}

#[derive(Clone, Debug)]
struct FunctionCallState {
    call_id: String,
    name: String,
    saw_arguments: bool,
    emitted: bool,
}

fn take_sse_block(buffer: &mut String) -> Option<String> {
    buffer.find("\n\n").map(|position| {
        let block = buffer[..position].to_string();
        *buffer = buffer[position + 2..].to_string();
        block
    })
}

fn extract_sse_data(block: &str) -> String {
    block
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|line| line.trim_start())
        .collect::<Vec<_>>()
        .join("\n")
}

fn ensure_message_start(
    message_started: &mut bool,
    response_id: Option<&str>,
    anthropic_model: &str,
) -> Option<String> {
    if *message_started {
        return None;
    }

    *message_started = true;
    Some(render_event(
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": response_id.unwrap_or("msg_proxy"),
                "type": "message",
                "role": "assistant",
                "model": anthropic_model,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            }
        }),
    ))
}

fn switch_to_text(current_block: &mut Option<BlockKind>, content_index: &mut usize) -> Vec<String> {
    switch_block(
        current_block,
        content_index,
        BlockKind::Text,
        json!({
            "type": "content_block_start",
            "index": *content_index,
            "content_block": {
                "type": "text",
                "text": "",
            }
        }),
    )
}

fn switch_to_thinking(
    current_block: &mut Option<BlockKind>,
    content_index: &mut usize,
) -> Vec<String> {
    switch_block(
        current_block,
        content_index,
        BlockKind::Thinking,
        json!({
            "type": "content_block_start",
            "index": *content_index,
            "content_block": {
                "type": "thinking",
                "thinking": "",
            }
        }),
    )
}

fn switch_to_tool_use(
    current_block: &mut Option<BlockKind>,
    content_index: &mut usize,
    item_id: String,
    call_id: &str,
    name: &str,
) -> Vec<String> {
    switch_block(
        current_block,
        content_index,
        BlockKind::ToolUse {
            item_id: item_id.clone(),
        },
        json!({
            "type": "content_block_start",
            "index": *content_index,
            "content_block": {
                "type": "tool_use",
                "id": call_id,
                "name": name,
            }
        }),
    )
}

fn switch_block(
    current_block: &mut Option<BlockKind>,
    content_index: &mut usize,
    next_block: BlockKind,
    start_payload: serde_json::Value,
) -> Vec<String> {
    if matches_same_block(current_block.as_ref(), &next_block) {
        return Vec::new();
    }

    let mut events = Vec::new();
    if let Some(stop) = close_current_block(current_block, content_index) {
        events.push(stop);
    }
    *current_block = Some(next_block);
    events.push(render_event("content_block_start", start_payload));
    events
}

fn matches_same_block(current: Option<&BlockKind>, next: &BlockKind) -> bool {
    match (current, next) {
        (Some(BlockKind::Text), BlockKind::Text) => true,
        (Some(BlockKind::Thinking), BlockKind::Thinking) => true,
        (
            Some(BlockKind::ToolUse { item_id: left }),
            BlockKind::ToolUse { item_id: right },
        ) => left == right,
        _ => false,
    }
}

fn close_current_block(
    current_block: &mut Option<BlockKind>,
    content_index: &mut usize,
) -> Option<String> {
    current_block.take().map(|_| {
        let event = render_event(
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": *content_index,
            }),
        );
        *content_index += 1;
        event
    })
}

fn message_delta_event(
    stop_reason: Option<String>,
    usage: Option<&responses::Usage>,
) -> String {
    render_event(
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": serde_json::Value::Null,
            },
            "usage": usage.map(|usage| json!({
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            })),
        }),
    )
}

fn error_event(error_type: &str, message: &str) -> String {
    render_event(
        "error",
        json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            }
        }),
    )
}

fn render_event(event_name: &str, payload: serde_json::Value) -> String {
    format!(
        "event: {}\ndata: {}\n\n",
        event_name,
        serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string())
    )
}

fn truncate_resp_id(id: &str) -> String {
    match id.strip_prefix("resp_") {
        Some(rest) if rest.len() > 10 => format!("resp_{}...", &rest[..10]),
        _ => id.to_string(),
    }
}