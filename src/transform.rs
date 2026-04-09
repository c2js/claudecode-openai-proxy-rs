use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::models::{anthropic, responses};
use serde_json::{json, Value};
use std::collections::HashMap;

const SUPPORTED_ANTHROPIC_MODELS: [&str; 3] = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
];

pub struct PreparedRequest {
    pub upstream_request: responses::ResponsesRequest,
    pub anthropic_model: String,
    pub upstream_model: String,
}

pub fn anthropic_to_responses(
    req: anthropic::AnthropicRequest,
    config: &Config,
) -> ProxyResult<PreparedRequest> {
    if req.top_k.is_some() {
        return Err(ProxyError::Transform(
            "Anthropic top_k is not supported by the OpenAI Responses API".to_string(),
        ));
    }

    if req.stop_sequences.as_ref().is_some_and(|stops| !stops.is_empty()) {
        return Err(ProxyError::Transform(
            "Anthropic stop_sequences are not supported by the OpenAI Responses API"
                .to_string(),
        ));
    }

    let anthropic_model = req.model.clone();
    let normalized_model = normalize_anthropic_model(&req.model)?;
    let upstream_model = map_upstream_model(config, &normalized_model)?;
    let reasoning = build_reasoning(&req, &upstream_model);

    if let Some(tool_choice) = req.tool_choice.as_ref() {
        if reasoning.is_some()
            && matches!(
                tool_choice,
                anthropic::ToolChoice::Any { .. } | anthropic::ToolChoice::Tool { .. }
            )
        {
            return Err(ProxyError::Transform(
                "Anthropic tool_choice any/tool is incompatible with thinking-enabled requests"
                    .to_string(),
            ));
        }
    }

    let (tool_choice, parallel_tool_calls) =
        map_tool_choice(req.tool_choice.as_ref(), req.tools.is_some());
    let text = build_text_config(&req);
    let metadata = metadata_to_strings(req.metadata.clone());
    let supports_sampling = should_forward_sampling(&upstream_model, reasoning.as_ref());

    let mut input = Vec::new();

    if let Some(system) = req.system {
        match system {
            anthropic::SystemPrompt::Single(text) => {
                input.push(message_item(
                    "system",
                    responses::ResponseMessageContent::Text(text),
                    None,
                ));
            }
            anthropic::SystemPrompt::Multiple(messages) => {
                for msg in messages {
                    input.push(message_item(
                        "system",
                        responses::ResponseMessageContent::Text(msg.text),
                        None,
                    ));
                }
            }
        }
    }

    for message in req.messages {
        input.extend(convert_message(message)?);
    }

    let tools = req.tools.map(convert_tools).transpose()?;

    Ok(PreparedRequest {
        anthropic_model,
        upstream_model: upstream_model.clone(),
        upstream_request: responses::ResponsesRequest {
            model: upstream_model,
            input,
            max_output_tokens: Some(req.max_tokens.max(16)),
            stream: req.stream,
            tools,
            tool_choice,
            parallel_tool_calls,
            reasoning,
            temperature: supports_sampling.then_some(req.temperature).flatten(),
            top_p: supports_sampling.then_some(req.top_p).flatten(),
            text,
            metadata,
            service_tier: req.service_tier,
            store: Some(false),
        },
    })
}

pub fn responses_to_anthropic(
    resp: responses::ResponsesResponse,
    anthropic_model: &str,
) -> ProxyResult<anthropic::AnthropicResponse> {
    if resp.status.as_deref() == Some("failed") {
        let message = resp
            .error
            .as_ref()
            .map(|error| error.message.clone())
            .unwrap_or_else(|| "Responses API request failed".to_string());
        return Err(ProxyError::Upstream(message));
    }

    let mut content = Vec::new();
    let mut saw_tool_use = false;

    for item in resp.output {
        match item {
            responses::ResponseOutputItem::Message { content: parts, .. } => {
                for part in parts {
                    match part {
                        responses::ResponseOutputContent::OutputText { text, .. } => {
                            if !text.is_empty() {
                                content.push(anthropic::ResponseContent::Text {
                                    content_type: "text".to_string(),
                                    text,
                                });
                            }
                        }
                        responses::ResponseOutputContent::Refusal { refusal } => {
                            if !refusal.is_empty() {
                                content.push(anthropic::ResponseContent::Text {
                                    content_type: "text".to_string(),
                                    text: refusal,
                                });
                            }
                        }
                    }
                }
            }
            responses::ResponseOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                let input = serde_json::from_str(&arguments).unwrap_or_else(|_| json!({}));
                content.push(anthropic::ResponseContent::ToolUse {
                    content_type: "tool_use".to_string(),
                    id: call_id,
                    name,
                    input,
                });
                saw_tool_use = true;
            }
            responses::ResponseOutputItem::Reasoning {
                content: reasoning_content,
                summary,
                ..
            } => {
                let mut thinking_parts: Vec<String> = summary
                    .into_iter()
                    .filter_map(|part| match part {
                        responses::ReasoningSummaryPart::SummaryText { text } if !text.is_empty() => {
                            Some(text)
                        }
                        _ => None,
                    })
                    .collect();

                if thinking_parts.is_empty() {
                    thinking_parts.extend(reasoning_content.into_iter().filter_map(|part| match part {
                        responses::ReasoningContent::ReasoningText { text } if !text.is_empty() => {
                            Some(text)
                        }
                        _ => None,
                    }));
                }

                for thinking in thinking_parts {
                    content.push(anthropic::ResponseContent::Thinking {
                        content_type: "thinking".to_string(),
                        thinking,
                    });
                }
            }
            responses::ResponseOutputItem::Unknown => {}
        }
    }

    if content.is_empty() {
        if let Some(output_text) = resp.output_text.filter(|text| !text.is_empty()) {
            content.push(anthropic::ResponseContent::Text {
                content_type: "text".to_string(),
                text: output_text,
            });
        }
    }

    let usage = resp.usage.unwrap_or(responses::Usage {
        input_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
        output_tokens_details: None,
    });

    Ok(anthropic::AnthropicResponse {
        id: resp.id,
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: anthropic_model.to_string(),
        stop_reason: map_responses_stop_reason(
            resp.status.as_deref(),
            resp.incomplete_details
                .as_ref()
                .map(|details| details.reason.as_str()),
            saw_tool_use,
        ),
        stop_sequence: None,
        usage: anthropic::Usage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
        },
    })
}

pub fn supported_models_response() -> anthropic::ModelsListResponse {
    let data = vec![
        anthropic::ModelInfo {
            created_at: "2025-10-01T00:00:00Z".to_string(),
            display_name: "Claude Haiku 4.5".to_string(),
            id: "claude-haiku-4-5".to_string(),
            model_type: "model".to_string(),
        },
        anthropic::ModelInfo {
            created_at: "2026-04-04T00:00:00Z".to_string(),
            display_name: "Claude Sonnet 4.6".to_string(),
            id: "claude-sonnet-4-6".to_string(),
            model_type: "model".to_string(),
        },
        anthropic::ModelInfo {
            created_at: "2026-04-04T00:00:00Z".to_string(),
            display_name: "Claude Opus 4.6".to_string(),
            id: "claude-opus-4-6".to_string(),
            model_type: "model".to_string(),
        },
    ];

    anthropic::ModelsListResponse {
        first_id: data.first().map(|model| model.id.clone()),
        last_id: data.last().map(|model| model.id.clone()),
        has_more: false,
        data,
    }
}

pub fn map_responses_stop_reason(
    status: Option<&str>,
    incomplete_reason: Option<&str>,
    saw_tool_use: bool,
) -> Option<String> {
    if saw_tool_use {
        return Some("tool_use".to_string());
    }

    match (status, incomplete_reason) {
        (Some("incomplete"), Some("max_output_tokens")) => Some("max_tokens".to_string()),
        (Some(_), _) | (None, _) => Some("end_turn".to_string()),
    }
}

pub fn normalize_anthropic_model(model: &str) -> ProxyResult<String> {
    let normalized = strip_date_suffix(model);
    if SUPPORTED_ANTHROPIC_MODELS.contains(&normalized.as_str()) {
        Ok(normalized)
    } else {
        Err(ProxyError::Transform(format!(
            "Unsupported Anthropic model '{}'. Supported models: {}",
            model,
            SUPPORTED_ANTHROPIC_MODELS.join(", ")
        )))
    }
}

pub fn is_allowed_upstream_model(model: &str) -> bool {
    model
        .split('/')
        .next_back()
        .map(|segment| segment.to_ascii_lowercase().starts_with("gpt-5"))
        .unwrap_or(false)
}

fn map_upstream_model(config: &Config, anthropic_model: &str) -> ProxyResult<String> {
    let candidate = config
        .model_map
        .get(anthropic_model)
        .cloned()
        .ok_or_else(|| {
            ProxyError::Config(format!(
                "No upstream model mapping configured for '{}'. Set MODEL_MAP to override the built-in defaults if needed.",
                anthropic_model
            ))
        })?;

    if is_allowed_upstream_model(&candidate) {
        Ok(candidate)
    } else {
        Err(ProxyError::Config(format!(
            "Mapped upstream model '{}' is not allowed. Only gpt-5 family models are supported.",
            candidate
        )))
    }
}

fn build_reasoning(
    req: &anthropic::AnthropicRequest,
    upstream_model: &str,
) -> Option<responses::ReasoningConfig> {
    let thinking_enabled = req
        .thinking
        .as_ref()
        .map(|thinking| matches!(thinking.thinking_type.as_str(), "enabled" | "adaptive"))
        .unwrap_or(false);

    let effort = req
        .output_config
        .as_ref()
        .and_then(|config| config.effort.clone())
        .map(|effort| map_effort_to_openai(&effort))
        .or_else(|| thinking_enabled.then(|| default_reasoning_effort(req)))
        .map(|effort| normalize_reasoning_effort(upstream_model, effort));

    effort.map(|effort| responses::ReasoningConfig {
        effort,
        summary: Some("auto".to_string()),
    })
}

fn normalize_reasoning_effort(upstream_model: &str, effort: String) -> String {
    if effort == "low" && is_gpt_5_4_pro_family(upstream_model) {
        "medium".to_string()
    } else {
        effort
    }
}

fn is_gpt_5_4_pro_family(model: &str) -> bool {
    model
        .split('/')
        .next_back()
        .map(|segment| segment.to_ascii_lowercase().starts_with("gpt-5.4-pro"))
        .unwrap_or(false)
}

fn default_reasoning_effort(req: &anthropic::AnthropicRequest) -> String {
    match req
        .thinking
        .as_ref()
        .and_then(|thinking| thinking.budget_tokens)
    {
        Some(0..=2048) => "low".to_string(),
        Some(2049..=8192) => "medium".to_string(),
        Some(8193..=32768) => "high".to_string(),
        Some(_) => "xhigh".to_string(),
        None => "high".to_string(),
    }
}

fn build_text_config(req: &anthropic::AnthropicRequest) -> Option<responses::ResponseTextConfig> {
    let format = req
        .output_config
        .as_ref()
        .and_then(|config| config.format.clone())
        .or_else(|| req.extra.get("output_format").cloned());

    format.map(|format| responses::ResponseTextConfig {
        format: Some(normalize_text_format(format)),
        verbosity: None,
    })
}

fn normalize_text_format(format: Value) -> Value {
    let mut format = format;

    let Some(object) = format.as_object_mut() else {
        return format;
    };

    if object.get("type").and_then(Value::as_str) != Some("json_schema") {
        return format;
    }

    if let Some(nested_json_schema) = object.remove("json_schema") {
        if let Some(nested_object) = nested_json_schema.as_object() {
            for (key, value) in nested_object {
                object.entry(key.clone()).or_insert_with(|| value.clone());
            }
        } else {
            object.insert("json_schema".to_string(), nested_json_schema);
        }
    }

    let sanitized_name = object
        .get("name")
        .and_then(Value::as_str)
        .map(sanitize_response_format_name)
        .filter(|name| !name.is_empty())
        .or_else(|| {
            object
                .get("schema")
                .and_then(Value::as_object)
                .and_then(|schema| schema.get("title"))
                .and_then(Value::as_str)
                .map(sanitize_response_format_name)
                .filter(|name| !name.is_empty())
        })
        .unwrap_or_else(|| "response".to_string());

    object.insert("name".to_string(), Value::String(sanitized_name));

    format
}

fn sanitize_response_format_name(name: &str) -> String {
    let sanitized = name
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => ch,
            _ => '_',
        })
        .take(64)
        .collect::<String>();

    sanitized.trim_matches('_').to_string()
}

fn metadata_to_strings(metadata: Option<Value>) -> Option<HashMap<String, String>> {
    let object = metadata?.as_object()?.clone();
    let converted = object
        .into_iter()
        .map(|(key, value)| {
            let value = match value {
                Value::String(text) => text,
                other => serde_json::to_string(&other).unwrap_or_else(|_| "null".to_string()),
            };
            (key, value)
        })
        .collect::<HashMap<_, _>>();

    (!converted.is_empty()).then_some(converted)
}

fn map_tool_choice(
    tool_choice: Option<&anthropic::ToolChoice>,
    has_tools: bool,
) -> (Option<responses::ResponseToolChoice>, Option<bool>) {
    match tool_choice {
        Some(anthropic::ToolChoice::Auto {
            disable_parallel_tool_use,
        }) => (
            Some(responses::ResponseToolChoice::Mode("auto".to_string())),
            disable_parallel_tool_use.map(|value| !value),
        ),
        Some(anthropic::ToolChoice::Any {
            disable_parallel_tool_use,
        }) => (
            Some(responses::ResponseToolChoice::Mode("required".to_string())),
            disable_parallel_tool_use.map(|value| !value),
        ),
        Some(anthropic::ToolChoice::Tool {
            name,
            disable_parallel_tool_use,
        }) => (
            Some(responses::ResponseToolChoice::Tool {
                tool_type: "function".to_string(),
                name: name.clone(),
            }),
            disable_parallel_tool_use.map(|value| !value),
        ),
        Some(anthropic::ToolChoice::None) => (
            Some(responses::ResponseToolChoice::Mode("none".to_string())),
            None,
        ),
        None if has_tools => (
            Some(responses::ResponseToolChoice::Mode("auto".to_string())),
            None,
        ),
        None => (None, None),
    }
}

fn should_forward_sampling(
    upstream_model: &str,
    reasoning: Option<&responses::ReasoningConfig>,
) -> bool {
    !is_allowed_upstream_model(upstream_model) || reasoning.is_none()
}

fn convert_tools(tools: Vec<anthropic::Tool>) -> ProxyResult<Vec<responses::ResponseTool>> {
    let mut converted = Vec::new();

    for tool in tools {
        if tool.tool_type.as_deref() == Some("BatchTool") {
            continue;
        }

        converted.push(responses::ResponseTool {
            tool_type: "function".to_string(),
            name: tool.name,
            description: tool.description,
            parameters: clean_schema(tool.input_schema),
            strict: tool.strict,
            defer_loading: tool.defer_loading,
        });
    }

    Ok(converted)
}

fn convert_message(message: anthropic::Message) -> ProxyResult<Vec<responses::ResponseInputItem>> {
    let mut items = Vec::new();

    match message.content {
        anthropic::MessageContent::Text(text) => {
            let phase = (message.role == "assistant").then_some("final_answer");
            items.push(message_item(
                message.role,
                responses::ResponseMessageContent::Text(text),
                phase,
            ));
        }
        anthropic::MessageContent::Blocks(blocks) => {
            let mut pending_parts = Vec::new();

            for block in blocks {
                match block {
                    anthropic::ContentBlock::Text { text, .. } => {
                        pending_parts.push(responses::ResponseInputContent::InputText { text });
                    }
                    anthropic::ContentBlock::Image { source } => {
                        if message.role != "user" {
                            return Err(ProxyError::Transform(
                                "Only user image inputs are supported when translating to the OpenAI Responses API"
                                    .to_string(),
                            ));
                        }

                        pending_parts.push(responses::ResponseInputContent::InputImage {
                            image_url: image_source_to_data_url(&source),
                            detail: None,
                        });
                    }
                    anthropic::ContentBlock::Thinking { thinking } => {
                        flush_pending_message(&mut items, &message.role, &mut pending_parts);
                        items.push(message_item(
                            message.role.clone(),
                            responses::ResponseMessageContent::Text(thinking),
                            (message.role == "assistant").then_some("commentary"),
                        ));
                    }
                    anthropic::ContentBlock::ToolUse { id, name, input } => {
                        flush_pending_message(&mut items, &message.role, &mut pending_parts);
                        items.push(responses::ResponseInputItem::FunctionCall {
                            call_id: id,
                            name,
                            arguments: serde_json::to_string(&input)
                                .map_err(ProxyError::Serialization)?,
                        });
                    }
                    anthropic::ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        flush_pending_message(&mut items, &message.role, &mut pending_parts);
                        items.push(responses::ResponseInputItem::FunctionCallOutput {
                            call_id: tool_use_id,
                            output: tool_result_to_output(content, is_error.unwrap_or(false)),
                        });
                    }
                }
            }

            flush_pending_message(&mut items, &message.role, &mut pending_parts);
        }
    }

    Ok(items)
}

fn flush_pending_message(
    items: &mut Vec<responses::ResponseInputItem>,
    role: &str,
    pending_parts: &mut Vec<responses::ResponseInputContent>,
) {
    if pending_parts.is_empty() {
        return;
    }

    let content = parts_to_message_content(std::mem::take(pending_parts));
    let phase = (role == "assistant").then_some("final_answer");
    items.push(message_item(role.to_string(), content, phase));
}

fn message_item(
    role: impl Into<String>,
    content: responses::ResponseMessageContent,
    phase: Option<&str>,
) -> responses::ResponseInputItem {
    responses::ResponseInputItem::Message {
        role: role.into(),
        content,
        phase: phase.map(str::to_string),
    }
}

fn parts_to_message_content(
    parts: Vec<responses::ResponseInputContent>,
) -> responses::ResponseMessageContent {
    if let [responses::ResponseInputContent::InputText { text }] = parts.as_slice() {
        responses::ResponseMessageContent::Text(text.clone())
    } else {
        responses::ResponseMessageContent::Parts(parts)
    }
}

fn tool_result_to_output(
    content: anthropic::ToolResultContent,
    is_error: bool,
) -> responses::FunctionCallOutput {
    match content {
        anthropic::ToolResultContent::Text(text) => {
            if is_error {
                responses::FunctionCallOutput::Text(
                    json!({ "is_error": true, "content": text }).to_string(),
                )
            } else {
                responses::FunctionCallOutput::Text(text)
            }
        }
        anthropic::ToolResultContent::Blocks(blocks) => {
            let mut parts = Vec::new();

            if is_error {
                parts.push(responses::ResponseInputContent::InputText {
                    text: "Tool returned an error.".to_string(),
                });
            }

            for block in blocks {
                match block {
                    anthropic::ToolResultBlock::Text { text } => {
                        parts.push(responses::ResponseInputContent::InputText { text });
                    }
                    anthropic::ToolResultBlock::Image { source } => {
                        parts.push(responses::ResponseInputContent::InputImage {
                            image_url: image_source_to_data_url(&source),
                            detail: None,
                        });
                    }
                }
            }

            responses::FunctionCallOutput::Parts(parts)
        }
    }
}

fn image_source_to_data_url(source: &anthropic::ImageSource) -> String {
    format!("data:{};base64,{}", source.media_type, source.data)
}

fn clean_schema(mut schema: Value) -> Value {
    if let Some(object) = schema.as_object_mut() {
        if object.get("format").and_then(Value::as_str) == Some("uri") {
            object.remove("format");
        }

        if let Some(properties) = object.get_mut("properties").and_then(Value::as_object_mut) {
            for value in properties.values_mut() {
                *value = clean_schema(value.clone());
            }
        }

        if let Some(items) = object.get_mut("items") {
            *items = clean_schema(items.clone());
        }
    }

    schema
}

fn strip_date_suffix(model: &str) -> String {
    if model.len() > 9 {
        let candidate = &model[model.len() - 9..];
        if candidate.starts_with('-') && candidate[1..].chars().all(|ch| ch.is_ascii_digit()) {
            return model[..model.len() - 9].to_string();
        }
    }

    model.to_string()
}

fn map_effort_to_openai(claude_effort: &str) -> String {
    match claude_effort {
        "low" => "low".to_string(),
        "medium" => "medium".to_string(),
        "high" => "high".to_string(),
        "max" => "xhigh".to_string(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        anthropic_to_responses, is_allowed_upstream_model, map_responses_stop_reason,
        normalize_anthropic_model, responses_to_anthropic,
    };
    use crate::{
        config::Config,
        models::{anthropic, responses},
    };
    use serde_json::json;
    use std::collections::HashMap;

    fn test_config() -> Config {
        Config {
            port: 18080,
            base_url: "https://api.openai.com".to_string(),
            api_key: None,
            model_map: HashMap::from([
                ("claude-haiku-4-5".to_string(), "gpt-5-mini".to_string()),
                ("claude-sonnet-4-6".to_string(), "gpt-5.4".to_string()),
                ("claude-opus-4-6".to_string(), "gpt-5.4-pro".to_string()),
            ]),
            debug: false,
            verbose: false,
            azure_openai_endpoint: None,
            azure_use_cli_credential: false,
        }
    }

    fn request(model: &str) -> anthropic::AnthropicRequest {
        anthropic::AnthropicRequest {
            model: model.to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("ping".to_string()),
            }],
            max_tokens: 512,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(false),
            tools: None,
            tool_choice: None,
            metadata: None,
            thinking: None,
            output_config: None,
            service_tier: None,
            extra: HashMap::new(),
        }
    }

    #[test]
    fn supported_models_accept_snapshot_suffixes() {
        assert_eq!(
            normalize_anthropic_model("claude-haiku-4-5-20251001").unwrap(),
            "claude-haiku-4-5"
        );
        assert_eq!(
            normalize_anthropic_model("claude-sonnet-4-6-20250514").unwrap(),
            "claude-sonnet-4-6"
        );
        assert!(normalize_anthropic_model("claude-3-5-sonnet").is_err());
    }

    #[test]
    fn upstream_models_are_limited_to_gpt5_family() {
        assert!(is_allowed_upstream_model("gpt-5.4"));
        assert!(is_allowed_upstream_model("openai/gpt-5.4-pro"));
        assert!(!is_allowed_upstream_model("gpt-4.1"));
    }

    #[test]
    fn anthropic_request_maps_model_and_tokens() {
        let prepared = anthropic_to_responses(request("claude-sonnet-4-6"), &test_config()).unwrap();

        assert_eq!(prepared.upstream_model, "gpt-5.4");
        assert_eq!(prepared.upstream_request.max_output_tokens, Some(512));
        assert_eq!(prepared.upstream_request.model, "gpt-5.4");
        assert_eq!(prepared.anthropic_model, "claude-sonnet-4-6");
    }

    #[test]
    fn anthropic_reasoning_maps_effort_and_disables_sampling() {
        let mut req = request("claude-opus-4-6");
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        req.thinking = Some(anthropic::ThinkingConfig {
            thinking_type: "adaptive".to_string(),
            budget_tokens: None,
        });
        req.output_config = Some(anthropic::OutputConfig {
            effort: Some("max".to_string()),
            format: None,
        });

        let prepared = anthropic_to_responses(req, &test_config()).unwrap();

        assert_eq!(prepared.upstream_request.reasoning.unwrap().effort, "xhigh");
        assert_eq!(prepared.upstream_request.temperature, None);
        assert_eq!(prepared.upstream_request.top_p, None);
    }

    #[test]
    fn low_effort_is_clamped_to_medium_for_gpt_5_4_pro() {
        let mut req = request("claude-opus-4-6");
        req.output_config = Some(anthropic::OutputConfig {
            effort: Some("low".to_string()),
            format: None,
        });

        let prepared = anthropic_to_responses(req, &test_config()).unwrap();

        assert_eq!(prepared.upstream_request.reasoning.unwrap().effort, "medium");
    }

    #[test]
    fn low_effort_stays_low_for_non_pro_gpt_5_models() {
        let mut req = request("claude-sonnet-4-6");
        req.output_config = Some(anthropic::OutputConfig {
            effort: Some("low".to_string()),
            format: None,
        });

        let prepared = anthropic_to_responses(req, &test_config()).unwrap();

        assert_eq!(prepared.upstream_request.reasoning.unwrap().effort, "low");
    }

    #[test]
    fn tool_use_and_tool_result_become_function_items() {
        let mut req = request("claude-sonnet-4-6");
        req.messages = vec![
            anthropic::Message {
                role: "assistant".to_string(),
                content: anthropic::MessageContent::Blocks(vec![
                    anthropic::ContentBlock::Text {
                        text: "Checking".to_string(),
                        cache_control: None,
                    },
                    anthropic::ContentBlock::ToolUse {
                        id: "call_1".to_string(),
                        name: "get_weather".to_string(),
                        input: json!({"city":"NYC"}),
                    },
                ]),
            },
            anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Blocks(vec![
                    anthropic::ContentBlock::ToolResult {
                        tool_use_id: "call_1".to_string(),
                        content: anthropic::ToolResultContent::Text("Sunny".to_string()),
                        is_error: None,
                    },
                ]),
            },
        ];

        let prepared = anthropic_to_responses(req, &test_config()).unwrap();

        assert!(matches!(
            prepared.upstream_request.input[0],
            responses::ResponseInputItem::Message { .. }
        ));
        assert!(matches!(
            prepared.upstream_request.input[1],
            responses::ResponseInputItem::FunctionCall { .. }
        ));
        assert!(matches!(
            prepared.upstream_request.input[2],
            responses::ResponseInputItem::FunctionCallOutput { .. }
        ));
    }

    #[test]
    fn output_format_maps_to_text_format() {
        let mut req = request("claude-sonnet-4-6");
        req.output_config = Some(anthropic::OutputConfig {
            effort: None,
            format: Some(json!({
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object"}
            })),
        });

        let prepared = anthropic_to_responses(req, &test_config()).unwrap();
        let format = prepared
            .upstream_request
            .text
            .and_then(|text| text.format)
            .expect("expected text.format");

        assert_eq!(format.get("type").and_then(|value| value.as_str()), Some("json_schema"));
        assert_eq!(format.get("name").and_then(|value| value.as_str()), Some("answer"));
    }

    #[test]
    fn output_format_without_name_gets_default_name() {
        let mut req = request("claude-haiku-4-5");
        req.output_config = Some(anthropic::OutputConfig {
            effort: None,
            format: Some(json!({
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    }
                }
            })),
        });

        let prepared = anthropic_to_responses(req, &test_config()).unwrap();
        let format = prepared
            .upstream_request
            .text
            .and_then(|text| text.format)
            .expect("expected text.format");

        assert_eq!(format.get("name").and_then(|value| value.as_str()), Some("response"));
    }

    #[test]
    fn nested_json_schema_format_is_flattened() {
        let mut req = request("claude-opus-4-6");
        req.extra.insert(
            "output_format".to_string(),
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "Claude Code Output",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reply": {"type": "string"}
                        }
                    }
                }
            }),
        );

        let prepared = anthropic_to_responses(req, &test_config()).unwrap();
        let format = prepared
            .upstream_request
            .text
            .and_then(|text| text.format)
            .expect("expected text.format");

        assert_eq!(
            format.get("name").and_then(|value| value.as_str()),
            Some("Claude_Code_Output")
        );
        assert_eq!(format.get("strict").and_then(|value| value.as_bool()), Some(true));
        assert!(format.get("schema").is_some());
        assert!(format.get("json_schema").is_none());
    }

    #[test]
    fn responses_output_maps_back_to_anthropic_blocks() {
        let response = responses::ResponsesResponse {
            id: "resp_1".to_string(),
            object: Some("response".to_string()),
            created_at: None,
            model: Some("gpt-5.4".to_string()),
            status: Some("completed".to_string()),
            output: vec![
                responses::ResponseOutputItem::Reasoning {
                    id: "rs_1".to_string(),
                    content: vec![],
                    summary: vec![responses::ReasoningSummaryPart::SummaryText {
                        text: "Thinking summary".to_string(),
                    }],
                },
                responses::ResponseOutputItem::Message {
                    id: "msg_1".to_string(),
                    role: "assistant".to_string(),
                    status: Some("completed".to_string()),
                    content: vec![responses::ResponseOutputContent::OutputText {
                        text: "Final answer".to_string(),
                        annotations: vec![],
                    }],
                },
                responses::ResponseOutputItem::FunctionCall {
                    id: "fc_1".to_string(),
                    call_id: "call_1".to_string(),
                    name: "tool_a".to_string(),
                    arguments: "{\"x\":1}".to_string(),
                    status: Some("completed".to_string()),
                },
            ],
            output_text: None,
            incomplete_details: None,
            usage: Some(responses::Usage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
                output_tokens_details: None,
            }),
            error: None,
        };

        let anthropic = responses_to_anthropic(response, "claude-sonnet-4-6").unwrap();

        assert_eq!(anthropic.model, "claude-sonnet-4-6");
        assert_eq!(anthropic.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(anthropic.usage.input_tokens, 10);
        assert_eq!(anthropic.content.len(), 3);
    }

    #[test]
    fn incomplete_responses_map_to_max_tokens() {
        assert_eq!(
            map_responses_stop_reason(Some("incomplete"), Some("max_output_tokens"), false)
                .as_deref(),
            Some("max_tokens")
        );
    }
}