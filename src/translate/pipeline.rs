use crate::error::{ProxyError, ProxyResult};
use crate::models::{anthropic, openai};
use crate::translate::core;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub struct TranslationPolicy {
    pub reasoning_model: Option<String>,
    pub completion_model: Option<String>,
    pub model_map: BTreeMap<String, String>,
    pub ignore_terms: Vec<String>,
}

pub fn translate_request(
    req: anthropic::AnthropicRequest,
    policy: &TranslationPolicy,
) -> ProxyResult<openai::OpenAIRequest> {
    let model = select_model(&req, policy);

    let mut openai_messages = Vec::new();

    if let Some(system) = req.system {
        let texts = match system {
            anthropic::SystemPrompt::Single(text) => vec![text],
            anthropic::SystemPrompt::Multiple(messages) => {
                messages.into_iter().map(|m| m.text).collect()
            }
        };
        for text in texts {
            openai_messages.push(openai::Message {
                role: "system".to_string(),
                content: Some(openai::MessageContent::Text(sanitize_prompt(
                    text,
                    &policy.ignore_terms,
                ))),
                ..Default::default()
            });
        }
    }

    for msg in req.messages {
        openai_messages.extend(core::translate_message(msg)?);
    }

    let tools = req.tools.and_then(|tools| {
        let filtered: Vec<_> = tools
            .into_iter()
            .filter(|t| !core::is_batch_tool(t))
            .map(core::translate_tool)
            .collect();

        if filtered.is_empty() {
            None
        } else {
            Some(filtered)
        }
    });

    // tool_choice is only meaningful when tools are present; otherwise upstream rejects it.
    let (tool_choice, parallel_tool_calls) = match (&tools, &req.tool_choice) {
        (Some(_), Some(choice)) => core::translate_tool_choice(choice),
        _ => (None, None),
    };

    let user = req.metadata.as_ref().and_then(extract_user_id);

    Ok(openai::OpenAIRequest {
        model,
        messages: openai_messages,
        max_tokens: Some(req.max_tokens),
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop_sequences,
        stream: req.stream,
        stream_options: req.stream.and_then(|stream| {
            stream.then_some(openai::StreamOptions {
                include_usage: true,
            })
        }),
        tools,
        tool_choice,
        parallel_tool_calls,
        user,
    })
}

/// Extract `metadata.user_id` (the only metadata field with an OpenAI equivalent).
fn extract_user_id(metadata: &Value) -> Option<String> {
    metadata
        .get("user_id")
        .and_then(Value::as_str)
        .map(String::from)
}

pub fn translate_response(
    resp: openai::OpenAIResponse,
    fallback_model: &str,
) -> ProxyResult<anthropic::AnthropicResponse> {
    let choice = resp
        .choices
        .first()
        .ok_or_else(|| ProxyError::Transform("No choices in response".to_string()))?;

    let mut content = Vec::new();

    // Thinking precedes the visible answer, matching Anthropic's block ordering.
    if let Some(reasoning) = &choice.message.reasoning_content {
        if !reasoning.is_empty() {
            content.push(anthropic::ResponseContent::Thinking {
                content_type: "thinking".to_string(),
                thinking: reasoning.clone(),
            });
        }
    }

    if let Some(text) = &choice.message.content {
        if !text.is_empty() {
            content.push(anthropic::ResponseContent::Text {
                content_type: "text".to_string(),
                text: text.clone(),
            });
        }
    }

    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            let input: Value =
                serde_json::from_str(&tool_call.function.arguments).unwrap_or_else(|_| json!({}));

            content.push(anthropic::ResponseContent::ToolUse {
                content_type: "tool_use".to_string(),
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                input,
            });
        }
    }

    let stop_reason = core::map_stop_reason(choice.finish_reason.as_deref());
    let (input_tokens, cache_read_input_tokens) = core::split_prompt_tokens(&resp.usage);

    Ok(anthropic::AnthropicResponse {
        id: resp.id.unwrap_or_else(|| "msg_proxy".to_string()),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: resp.model.unwrap_or_else(|| fallback_model.to_string()),
        stop_reason,
        stop_sequence: None,
        usage: anthropic::Usage {
            input_tokens,
            output_tokens: resp.usage.completion_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens,
        },
    })
}

/// Per-message chat-template overhead (e.g. qwen's `<|im_start|>role\n…<|im_end|>\n`
/// plus the trailing assistant primer) in tokens. Measured at 8–10 across qwen models;
/// we use the high end so the count is never below the upstream's real count.
const PER_MESSAGE_OVERHEAD: usize = 10;

/// Count the input tokens for a count-tokens request using a real BPE tokenizer
/// (see [`estimate_text_tokens`]) plus small per-message overhead. Accurate token counts
/// are what let Claude Code decide compaction at the right time; a rough heuristic
/// under-counted code and made it compact too late. Base64 image payloads are excluded.
pub fn estimate_input_tokens(req: &anthropic::CountTokensRequest) -> u32 {
    let mut tokens = 0usize;

    if let Some(system) = &req.system {
        match system {
            anthropic::SystemPrompt::Single(text) => tokens += estimate_text_tokens(text),
            anthropic::SystemPrompt::Multiple(messages) => {
                for msg in messages {
                    tokens += estimate_text_tokens(&msg.text);
                }
            }
        }
    }

    for msg in &req.messages {
        tokens += estimate_text_tokens(&msg.role);
        tokens += estimate_content_tokens(&msg.content);
        tokens += PER_MESSAGE_OVERHEAD;
    }

    if let Some(tools) = &req.tools {
        for tool in tools {
            tokens += estimate_text_tokens(&tool.name);
            if let Some(description) = &tool.description {
                tokens += estimate_text_tokens(description);
            }
            tokens += estimate_text_tokens(&tool.input_schema.to_string());
        }
    }

    // cl100k slightly under-counts the qwen-class upstreams on dense code (measured ~11%);
    // a 1.15× safety factor keeps our count at or above the real one so Claude Code never
    // under-estimates and compacts on time. Over-counting only triggers compaction a touch
    // early — far cheaper than overflowing the window.
    ((tokens * 23 / 20).max(1)) as u32
}

/// Heuristic input-token estimate for an already-translated OpenAI request. Used to fill
/// `input_tokens` in the streaming `message_start` — the upstream only reports real usage
/// in the final chunk (too late for message_start), but Claude Code reads input from
/// message_start to track context. Mirrors [`estimate_input_tokens`], safety factor included.
pub fn estimate_openai_input_tokens(req: &openai::OpenAIRequest) -> u32 {
    let mut tokens = 0usize;

    for msg in &req.messages {
        tokens += estimate_text_tokens(&msg.role);
        match &msg.content {
            Some(openai::MessageContent::Text(text)) => tokens += estimate_text_tokens(text),
            Some(openai::MessageContent::Parts(parts)) => {
                for part in parts {
                    if let openai::ContentPart::Text { text } = part {
                        tokens += estimate_text_tokens(text);
                    }
                }
            }
            None => {}
        }
        if let Some(reasoning) = &msg.reasoning_content {
            tokens += estimate_text_tokens(reasoning);
        }
        if let Some(tool_calls) = &msg.tool_calls {
            for call in tool_calls {
                tokens += estimate_text_tokens(&call.function.name);
                tokens += estimate_text_tokens(&call.function.arguments);
            }
        }
        tokens += PER_MESSAGE_OVERHEAD;
    }

    if let Some(tools) = &req.tools {
        for tool in tools {
            tokens += estimate_text_tokens(&tool.function.name);
            if let Some(description) = &tool.function.description {
                tokens += estimate_text_tokens(description);
            }
            tokens += estimate_text_tokens(&tool.function.parameters.to_string());
        }
    }

    ((tokens * 23 / 20).max(1)) as u32
}

/// Concatenate all countable text (system + message content + tool schemas) into a
/// single string for an upstream `/tokenize` call, returning it with the message count
/// (used to add the per-message chat-template overhead). Base64 images are excluded.
pub fn collect_tokenize_text(req: &anthropic::CountTokensRequest) -> (String, usize) {
    let mut parts: Vec<String> = Vec::new();

    if let Some(system) = &req.system {
        match system {
            anthropic::SystemPrompt::Single(text) => parts.push(text.clone()),
            anthropic::SystemPrompt::Multiple(messages) => {
                parts.extend(messages.iter().map(|m| m.text.clone()));
            }
        }
    }

    for msg in &req.messages {
        parts.push(msg.role.clone());
        collect_content_text(&msg.content, &mut parts);
    }

    if let Some(tools) = &req.tools {
        for tool in tools {
            parts.push(tool.name.clone());
            if let Some(description) = &tool.description {
                parts.push(description.clone());
            }
            parts.push(tool.input_schema.to_string());
        }
    }

    (parts.join("\n"), req.messages.len())
}

fn collect_content_text(content: &anthropic::MessageContent, parts: &mut Vec<String>) {
    match content {
        anthropic::MessageContent::Text(text) => parts.push(text.clone()),
        anthropic::MessageContent::Blocks(blocks) => {
            for block in blocks {
                match block {
                    anthropic::ContentBlock::Text { text, .. } => parts.push(text.clone()),
                    anthropic::ContentBlock::Thinking { thinking } => parts.push(thinking.clone()),
                    anthropic::ContentBlock::ToolUse { name, input, .. } => {
                        parts.push(name.clone());
                        parts.push(input.to_string());
                    }
                    anthropic::ContentBlock::ToolResult { content, .. } => match content {
                        anthropic::ToolResultContent::Text(text) => parts.push(text.clone()),
                        anthropic::ToolResultContent::Blocks(blocks) => {
                            for b in blocks {
                                if let anthropic::ContentBlock::Text { text, .. } = b {
                                    parts.push(text.clone());
                                }
                            }
                        }
                    },
                    anthropic::ContentBlock::Image { .. } => {}
                }
            }
        }
    }
}

/// Per-message chat-template overhead in tokens, added on top of an exact upstream
/// content count (and used by the heuristic estimate too).
pub const PER_MESSAGE_OVERHEAD_TOKENS: u32 = PER_MESSAGE_OVERHEAD as u32;

/// Lazily-initialized BPE tokenizer (cl100k_base). Shared across requests.
fn tokenizer() -> Option<&'static tiktoken_rs::CoreBPE> {
    static BPE: std::sync::OnceLock<Option<tiktoken_rs::CoreBPE>> = std::sync::OnceLock::new();
    BPE.get_or_init(|| tiktoken_rs::cl100k_base().ok()).as_ref()
}

/// Per-text token count via the cl100k BPE tokenizer — the same family the OpenAI
/// ecosystem uses, far closer to reality than the old char heuristic that under-counted
/// code (and made Claude Code compact too late and overflow). cl100k still runs ~10%
/// under the qwen tokenizer on dense code, which the caller's safety factor offsets.
/// This path is only the fallback for when the upstream `/tokenize` is unavailable.
fn estimate_text_tokens(text: &str) -> usize {
    match tokenizer() {
        Some(bpe) => bpe.encode_ordinary(text).len(),
        // cl100k is embedded and effectively never fails to load; this only guards a
        // corrupt build, so a crude byte heuristic is plenty.
        None => text.len() / 4,
    }
}

fn estimate_content_tokens(content: &anthropic::MessageContent) -> usize {
    match content {
        anthropic::MessageContent::Text(text) => estimate_text_tokens(text),
        anthropic::MessageContent::Blocks(blocks) => blocks.iter().map(estimate_block_tokens).sum(),
    }
}

fn estimate_block_tokens(block: &anthropic::ContentBlock) -> usize {
    match block {
        anthropic::ContentBlock::Text { text, .. } => estimate_text_tokens(text),
        anthropic::ContentBlock::Thinking { thinking } => estimate_text_tokens(thinking),
        anthropic::ContentBlock::ToolUse { name, input, .. } => {
            estimate_text_tokens(name) + estimate_text_tokens(&input.to_string())
        }
        anthropic::ContentBlock::ToolResult { content, .. } => match content {
            anthropic::ToolResultContent::Text(text) => estimate_text_tokens(text),
            anthropic::ToolResultContent::Blocks(blocks) => {
                blocks.iter().map(estimate_block_tokens).sum()
            }
        },
        // Image payloads (base64) are intentionally not counted — they would
        // wildly inflate the estimate and the upstream text models ignore them.
        anthropic::ContentBlock::Image { .. } => 0,
    }
}

pub fn translate_models_list(resp: openai::ModelsListResponse) -> anthropic::ModelsListResponse {
    let data: Vec<_> = resp
        .data
        .into_iter()
        .map(|model| anthropic::ModelInfo {
            created_at: "1970-01-01T00:00:00Z".to_string(),
            display_name: model.id.clone(),
            id: model.id,
            model_type: "model".to_string(),
        })
        .collect();

    let first_id = data.first().map(|m| m.id.clone());
    let last_id = data.last().map(|m| m.id.clone());

    anthropic::ModelsListResponse {
        data,
        first_id,
        has_more: false,
        last_id,
    }
}

fn select_model(req: &anthropic::AnthropicRequest, policy: &TranslationPolicy) -> String {
    let has_thinking = req
        .extra
        .get("thinking")
        .and_then(|v| v.as_object())
        .map(|o| o.get("type").and_then(|t| t.as_str()) == Some("enabled"))
        .unwrap_or(false);

    let model = if has_thinking {
        policy
            .reasoning_model
            .clone()
            .unwrap_or_else(|| req.model.clone())
    } else {
        policy
            .completion_model
            .clone()
            .unwrap_or_else(|| req.model.clone())
    };

    policy.model_map.get(&model).cloned().unwrap_or(model)
}

fn sanitize_prompt(text: String, terms: &[String]) -> String {
    let mut sanitized = text;
    let mut removed = Vec::new();

    for term in terms {
        let next = core::remove_term(&sanitized, term);
        if next != sanitized {
            sanitized = next;
            removed.push(term.clone());
        }
    }

    if !removed.is_empty() {
        tracing::debug!(
            "Removed configured system prompt terms for upstream compatibility: {}",
            removed.join("; ")
        );
    }

    sanitized
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use serde_json::json;

    fn policy_from(config: &Config) -> TranslationPolicy {
        TranslationPolicy {
            reasoning_model: config.reasoning_model.clone(),
            completion_model: config.completion_model.clone(),
            model_map: config.model_map.clone(),
            ignore_terms: config.system_prompt_ignore_terms.clone(),
        }
    }

    fn default_policy() -> TranslationPolicy {
        policy_from(&Config::default())
    }

    #[test]
    fn applies_model_map_after_selection() {
        let req = anthropic::AnthropicRequest {
            model: "claude-opus-4-6".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("pong".to_string()),
            }],
            max_tokens: 64,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(false),
            tools: None,
            metadata: None,
            tool_choice: None,
            extra: json!({}),
        };

        let policy = TranslationPolicy {
            model_map: [("claude-opus-4-6".to_string(), "openai/gpt-4.1".to_string())]
                .into_iter()
                .collect(),
            ..default_policy()
        };

        let openai = translate_request(req, &policy).unwrap();
        assert_eq!(openai.model, "openai/gpt-4.1");
    }

    #[test]
    fn sanitizes_configured_system_prompt_terms() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("pong".to_string()),
            }],
            max_tokens: 64,
            system: Some(anthropic::SystemPrompt::Single(
                "Examples of risky actions: rm -rf.".to_string(),
            )),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(true),
            tools: None,
            metadata: None,
            tool_choice: None,
            extra: json!({}),
        };

        let policy = TranslationPolicy {
            ignore_terms: vec!["rm -rf".to_string()],
            ..default_policy()
        };

        let openai = translate_request(req, &policy).unwrap();

        match &openai.messages[0].content {
            Some(openai::MessageContent::Text(text)) => {
                assert_eq!(text, "Examples of risky actions: .");
            }
            _ => panic!("expected sanitized system prompt"),
        }
    }

    #[test]
    fn streaming_request_includes_usage_stream_options() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(true),
            tools: None,
            metadata: None,
            tool_choice: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        assert_eq!(
            openai.stream_options.map(|options| options.include_usage),
            Some(true)
        );
    }

    #[test]
    fn non_streaming_request_omits_usage_stream_options() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(false),
            tools: None,
            metadata: None,
            tool_choice: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        assert!(openai.stream_options.is_none());
    }

    #[test]
    fn converts_tool_definitions() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("use tool".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: Some(vec![anthropic::Tool {
                name: "read_file".to_string(),
                description: Some("Read a file".to_string()),
                input_schema: json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
                tool_type: None,
            }]),
            metadata: None,
            tool_choice: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        let tools = openai.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "read_file");
    }

    #[test]
    fn filters_batch_tools() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: Some(vec![anthropic::Tool {
                name: "batch_tool".to_string(),
                description: None,
                input_schema: json!({}),
                tool_type: Some("BatchTool".to_string()),
            }]),
            metadata: None,
            tool_choice: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();
        assert!(openai.tools.is_none());
    }

    #[test]
    fn converts_image_content() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Blocks(vec![
                    anthropic::ContentBlock::Text {
                        text: "What is this?".to_string(),
                        cache_control: None,
                    },
                    anthropic::ContentBlock::Image {
                        source: anthropic::ImageSource {
                            source_type: "base64".to_string(),
                            media_type: "image/png".to_string(),
                            data: "iVBOR...".to_string(),
                        },
                    },
                ]),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            tool_choice: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        match &openai.messages[0].content {
            Some(openai::MessageContent::Parts(parts)) => {
                assert_eq!(parts.len(), 2);
                match &parts[1] {
                    openai::ContentPart::ImageUrl { image_url } => {
                        assert!(image_url.url.starts_with("data:image/png;base64,"));
                    }
                    _ => panic!("expected image_url part"),
                }
            }
            _ => panic!("expected multi-part content"),
        }
    }

    #[test]
    fn converts_tool_use_and_tool_result() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                anthropic::Message {
                    role: "assistant".to_string(),
                    content: anthropic::MessageContent::Blocks(vec![
                        anthropic::ContentBlock::ToolUse {
                            id: "tool_1".to_string(),
                            name: "read_file".to_string(),
                            input: json!({"path": "/tmp"}),
                        },
                    ]),
                },
                anthropic::Message {
                    role: "user".to_string(),
                    content: anthropic::MessageContent::Blocks(vec![
                        anthropic::ContentBlock::ToolResult {
                            tool_use_id: "tool_1".to_string(),
                            content: anthropic::ToolResultContent::Text(
                                "file contents".to_string(),
                            ),
                            is_error: None,
                        },
                    ]),
                },
            ],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            tool_choice: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        let tool_calls = openai.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls[0].id, "tool_1");
        assert_eq!(tool_calls[0].function.name, "read_file");

        assert_eq!(openai.messages[1].role, "tool");
        assert_eq!(openai.messages[1].tool_call_id, Some("tool_1".to_string()));
    }

    #[test]
    fn deserializes_tool_result_with_nested_content_blocks() {
        let body = json!({
            "model": "gpt-4o",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tool_42",
                    "content": [
                        {"type": "text", "text": "first chunk"},
                        {"type": "text", "text": "second chunk"}
                    ]
                }]
            }]
        });

        let req: anthropic::AnthropicRequest = serde_json::from_value(body).unwrap();
        let openai = translate_request(req, &default_policy()).unwrap();

        let tool_msg = openai
            .messages
            .iter()
            .find(|m| m.role == "tool")
            .expect("expected a tool message");
        assert_eq!(tool_msg.tool_call_id, Some("tool_42".to_string()));
        match &tool_msg.content {
            Some(openai::MessageContent::Text(text)) => {
                assert_eq!(text, "first chunk\nsecond chunk");
            }
            other => panic!("expected flattened text content, got {:?}", other),
        }
    }

    #[test]
    fn converts_multiple_system_prompts() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 100,
            system: Some(anthropic::SystemPrompt::Multiple(vec![
                anthropic::SystemMessage {
                    message_type: "text".to_string(),
                    text: "You are helpful.".to_string(),
                    cache_control: None,
                },
                anthropic::SystemMessage {
                    message_type: "text".to_string(),
                    text: "Be concise.".to_string(),
                    cache_control: None,
                },
            ])),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            tool_choice: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        let system_msgs: Vec<_> = openai
            .messages
            .iter()
            .filter(|m| m.role == "system")
            .collect();
        assert_eq!(system_msgs.len(), 2);
    }

    #[test]
    fn uses_reasoning_model_when_thinking_enabled() {
        let req = anthropic::AnthropicRequest {
            model: "claude-opus-4-6".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("think hard".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            tool_choice: None,
            extra: json!({"thinking": {"type": "enabled", "budget_tokens": 1000}}),
        };

        let policy = TranslationPolicy {
            reasoning_model: Some("gpt-4o-reasoning".to_string()),
            completion_model: Some("gpt-4o-mini".to_string()),
            ..default_policy()
        };

        let openai = translate_request(req, &policy).unwrap();
        assert_eq!(openai.model, "gpt-4o-reasoning");
    }

    #[test]
    fn uses_completion_model_without_thinking() {
        let req = anthropic::AnthropicRequest {
            model: "claude-opus-4-6".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("quick".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            tool_choice: None,
            extra: json!({}),
        };

        let policy = TranslationPolicy {
            reasoning_model: Some("gpt-4o-reasoning".to_string()),
            completion_model: Some("gpt-4o-mini".to_string()),
            ..default_policy()
        };

        let openai = translate_request(req, &policy).unwrap();
        assert_eq!(openai.model, "gpt-4o-mini");
    }

    #[test]
    fn response_with_all_fields_present() {
        let response = openai::OpenAIResponse {
            id: Some("chatcmpl-abc123".to_string()),
            object: Some("chat.completion".to_string()),
            created: Some(1700000000),
            model: Some("gpt-4o".to_string()),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChoiceMessage {
                    role: "assistant".to_string(),
                    reasoning_content: None,
                    content: Some("hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: openai::Usage {
                prompt_tokens: 5,
                completion_tokens: 1,
                total_tokens: 6,
                ..Default::default()
            },
            system_fingerprint: None,
        };

        let anthropic = translate_response(response, "fallback-model").unwrap();
        assert_eq!(anthropic.id, "chatcmpl-abc123");
        assert_eq!(anthropic.model, "gpt-4o");
    }

    #[test]
    fn response_allows_missing_metadata() {
        let response = openai::OpenAIResponse {
            id: None,
            object: None,
            created: None,
            model: None,
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChoiceMessage {
                    role: "assistant".to_string(),
                    reasoning_content: None,
                    content: Some("pong".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: openai::Usage {
                prompt_tokens: 10,
                completion_tokens: 2,
                total_tokens: 12,
                ..Default::default()
            },
            system_fingerprint: None,
        };

        let anthropic = translate_response(response, "openai/gpt-4o-mini").unwrap();
        assert_eq!(anthropic.id, "msg_proxy");
        assert_eq!(anthropic.model, "openai/gpt-4o-mini");
    }

    #[test]
    fn response_converts_tool_calls() {
        let response = openai::OpenAIResponse {
            id: Some("chatcmpl-1".to_string()),
            object: None,
            created: None,
            model: Some("gpt-4o".to_string()),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChoiceMessage {
                    role: "assistant".to_string(),
                    reasoning_content: None,
                    content: None,
                    tool_calls: Some(vec![openai::ToolCall {
                        id: "call_abc".to_string(),
                        call_type: "function".to_string(),
                        function: openai::FunctionCall {
                            name: "read_file".to_string(),
                            arguments: "{\"path\":\"/tmp\"}".to_string(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: openai::Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                ..Default::default()
            },
            system_fingerprint: None,
        };

        let anthropic = translate_response(response, "fallback").unwrap();
        assert_eq!(anthropic.stop_reason, Some("tool_use".to_string()));
        assert!(!anthropic.content.is_empty());
    }

    #[test]
    fn models_list_translation() {
        let response = openai::ModelsListResponse {
            object: Some("list".to_string()),
            data: vec![
                openai::ModelInfo {
                    id: "gpt-4o-mini".to_string(),
                    object: Some("model".to_string()),
                    created: None,
                    owned_by: Some("azure".to_string()),
                },
                openai::ModelInfo {
                    id: "gpt-5-chat".to_string(),
                    object: Some("model".to_string()),
                    created: None,
                    owned_by: Some("azure".to_string()),
                },
            ],
        };

        let result = translate_models_list(response);
        assert_eq!(result.first_id.as_deref(), Some("gpt-4o-mini"));
        assert_eq!(result.last_id.as_deref(), Some("gpt-5-chat"));
        assert!(!result.has_more);
    }

    #[test]
    fn empty_models_list() {
        let response = openai::ModelsListResponse {
            object: Some("list".to_string()),
            data: vec![],
        };
        let result = translate_models_list(response);
        assert!(result.data.is_empty());
        assert!(result.first_id.is_none());
    }

    fn base_request() -> anthropic::AnthropicRequest {
        anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 64,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            tool_choice: None,
            extra: json!({}),
        }
    }

    fn weather_tool() -> anthropic::Tool {
        anthropic::Tool {
            name: "get_weather".to_string(),
            description: None,
            input_schema: json!({"type": "object"}),
            tool_type: None,
        }
    }

    #[test]
    fn translates_tool_choice_when_tools_present() {
        let req = anthropic::AnthropicRequest {
            tools: Some(vec![weather_tool()]),
            tool_choice: Some(json!({"type": "tool", "name": "get_weather"})),
            ..base_request()
        };

        let openai = translate_request(req, &default_policy()).unwrap();
        assert_eq!(
            openai.tool_choice,
            Some(json!({"type": "function", "function": {"name": "get_weather"}}))
        );
    }

    #[test]
    fn drops_tool_choice_when_no_tools() {
        let req = anthropic::AnthropicRequest {
            tools: None,
            tool_choice: Some(json!({"type": "auto"})),
            ..base_request()
        };

        let openai = translate_request(req, &default_policy()).unwrap();
        assert_eq!(openai.tool_choice, None);
        assert_eq!(openai.parallel_tool_calls, None);
    }

    #[test]
    fn forwards_disable_parallel_tool_use() {
        let req = anthropic::AnthropicRequest {
            tools: Some(vec![weather_tool()]),
            tool_choice: Some(json!({"type": "auto", "disable_parallel_tool_use": true})),
            ..base_request()
        };

        let openai = translate_request(req, &default_policy()).unwrap();
        assert_eq!(openai.tool_choice, Some(json!("auto")));
        assert_eq!(openai.parallel_tool_calls, Some(false));
    }

    #[test]
    fn maps_metadata_user_id_to_user() {
        let req = anthropic::AnthropicRequest {
            metadata: Some(json!({"user_id": "user-123"})),
            ..base_request()
        };

        let openai = translate_request(req, &default_policy()).unwrap();
        assert_eq!(openai.user.as_deref(), Some("user-123"));
    }

    #[test]
    fn metadata_without_user_id_leaves_user_unset() {
        let req = anthropic::AnthropicRequest {
            metadata: Some(json!({"other": "value"})),
            ..base_request()
        };

        let openai = translate_request(req, &default_policy()).unwrap();
        assert_eq!(openai.user, None);
    }

    #[test]
    fn response_content_filter_maps_to_refusal() {
        let response = openai::OpenAIResponse {
            id: Some("chatcmpl-1".to_string()),
            object: None,
            created: None,
            model: Some("gpt-4o".to_string()),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChoiceMessage {
                    role: "assistant".to_string(),
                    reasoning_content: None,
                    content: Some("I can't help with that.".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("content_filter".to_string()),
            }],
            usage: openai::Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                ..Default::default()
            },
            system_fingerprint: None,
        };

        let anthropic = translate_response(response, "fallback").unwrap();
        assert_eq!(anthropic.stop_reason, Some("refusal".to_string()));
    }

    // --- Upstream response robustness (regression guards against 502s) ---

    fn parse_response(value: serde_json::Value) -> openai::OpenAIResponse {
        serde_json::from_value(value).expect("upstream response should deserialize")
    }

    #[test]
    fn response_parses_without_usage_field() {
        let resp = parse_response(json!({
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]
        }));
        let anthropic = translate_response(resp, "fallback").unwrap();
        assert_eq!(anthropic.usage.input_tokens, 0);
        assert_eq!(anthropic.usage.output_tokens, 0);
    }

    #[test]
    fn response_parses_with_null_usage() {
        let resp = parse_response(json!({
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
            "usage": null
        }));
        let anthropic = translate_response(resp, "fallback").unwrap();
        assert_eq!(anthropic.usage.input_tokens, 0);
    }

    #[test]
    fn response_parses_with_partial_usage() {
        // Missing total_tokens must not abort parsing.
        let resp = parse_response(json!({
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 11}
        }));
        let anthropic = translate_response(resp, "fallback").unwrap();
        assert_eq!(anthropic.usage.input_tokens, 11);
        assert_eq!(anthropic.usage.output_tokens, 0);
    }

    #[test]
    fn response_parses_with_missing_role() {
        let resp = parse_response(json!({
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "message": {"content": "hi"}}]
        }));
        let anthropic = translate_response(resp, "fallback").unwrap();
        assert_eq!(anthropic.content.len(), 1);
    }

    #[test]
    fn response_ignores_unknown_fields() {
        let resp = parse_response(json!({
            "id": "chatcmpl-1",
            "service_tier": "default",
            "choices": [{
                "index": 0,
                "logprobs": null,
                "message": {"role": "assistant", "content": "hi", "annotations": []}
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
                      "prompt_tokens_details": {"cached_tokens": 4}}
        }));
        let anthropic = translate_response(resp, "fallback").unwrap();
        // Cached tokens are split out: input = prompt(10) - cached(4), cache_read = 4.
        assert_eq!(anthropic.usage.input_tokens, 6);
        assert_eq!(anthropic.usage.output_tokens, 2);
        assert_eq!(anthropic.usage.cache_read_input_tokens, Some(4));
    }

    #[test]
    fn response_without_cache_details_omits_cache_fields() {
        let resp = parse_response(json!({
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
        }));
        let anthropic = translate_response(resp, "fallback").unwrap();
        assert_eq!(anthropic.usage.input_tokens, 7);
        assert_eq!(anthropic.usage.cache_read_input_tokens, None);
        assert_eq!(anthropic.usage.cache_creation_input_tokens, None);
    }

    #[test]
    fn response_reasoning_content_becomes_thinking_block() {
        let resp = parse_response(json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "reasoning_content": "step 1", "content": "answer"},
                "finish_reason": "stop"
            }]
        }));
        let anthropic = translate_response(resp, "fallback").unwrap();
        assert_eq!(anthropic.content.len(), 2);
        assert!(matches!(
            &anthropic.content[0],
            anthropic::ResponseContent::Thinking { thinking, .. } if thinking == "step 1"
        ));
        assert!(matches!(
            &anthropic.content[1],
            anthropic::ResponseContent::Text { text, .. } if text == "answer"
        ));
    }

    #[test]
    fn estimate_tokens_scales_with_text_and_is_nonzero() {
        let small: anthropic::CountTokensRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let large: anthropic::CountTokensRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "x".repeat(4000)}]
        }))
        .unwrap();
        assert!(estimate_input_tokens(&small) >= 1);
        assert!(estimate_input_tokens(&large) > estimate_input_tokens(&small));
    }

    #[test]
    fn estimate_matches_real_tokenizer_for_code() {
        // The whole point: a real BPE tokenizer counts code accurately, where the old
        // chars/4 heuristic under-counted (which is what made Claude Code compact too late).
        let code = "fn main() { let xs: Vec<i32> = (0..100).filter(|n| n % 2 == 0).collect(); \
                    println!(\"{:?}\", xs); }";
        let req: anthropic::CountTokensRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": code}]
        }))
        .unwrap();
        let est = estimate_input_tokens(&req) as usize;
        // cl100k tokenizes this to ~45 tokens; the old chars/4 would have said ~26.
        assert!(
            est > code.len() / 4,
            "estimate {est} should exceed the old chars/4"
        );
        assert!(est >= 30);
    }

    #[test]
    fn estimate_counts_cjk() {
        let cjk: anthropic::CountTokensRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "你好世界你好世界你好世界"}]
        }))
        .unwrap();
        // CJK tokenizes to multiple tokens per character — must be a meaningful count.
        assert!(estimate_input_tokens(&cjk) >= 12);
    }

    #[test]
    fn estimate_tokens_counts_system_and_tools_but_not_images() {
        let with_image: anthropic::CountTokensRequest = serde_json::from_value(json!({
            "system": "be terse",
            "tools": [{"name": "t", "description": "d", "input_schema": {"type": "object"}}],
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A".repeat(100000)}},
                    {"type": "text", "text": "tiny"}
                ]
            }]
        }))
        .unwrap();
        // A 100k-char base64 blob must NOT dominate the estimate.
        assert!(estimate_input_tokens(&with_image) < 200);
    }

    #[test]
    fn response_reasoning_alias_is_accepted() {
        let resp = parse_response(json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "reasoning": "thinking via alias", "content": ""},
                "finish_reason": "length"
            }]
        }));
        let anthropic = translate_response(resp, "fallback").unwrap();
        assert_eq!(anthropic.content.len(), 1);
        assert!(matches!(
            &anthropic.content[0],
            anthropic::ResponseContent::Thinking { thinking, .. } if thinking == "thinking via alias"
        ));
    }
}
