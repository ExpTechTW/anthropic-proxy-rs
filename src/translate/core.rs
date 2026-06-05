use crate::error::{ProxyError, ProxyResult};
use crate::models::{anthropic, openai};
use serde_json::Value;

pub fn translate_message(msg: anthropic::Message) -> ProxyResult<Vec<openai::Message>> {
    let mut result = Vec::new();

    match msg.content {
        anthropic::MessageContent::Text(text) => {
            result.push(openai::Message {
                role: msg.role,
                content: Some(openai::MessageContent::Text(text)),
                ..Default::default()
            });
        }
        anthropic::MessageContent::Blocks(blocks) => {
            let mut content_parts = Vec::new();
            let mut reasoning_parts = Vec::new();
            let mut tool_calls = Vec::new();

            for block in blocks {
                match block {
                    anthropic::ContentBlock::Text { text, .. } => {
                        content_parts.push(openai::ContentPart::Text { text });
                    }
                    anthropic::ContentBlock::Image { source } => {
                        let data_url = format!("data:{};base64,{}", source.media_type, source.data);
                        content_parts.push(openai::ContentPart::ImageUrl {
                            image_url: openai::ImageUrl { url: data_url },
                        });
                    }
                    anthropic::ContentBlock::ToolUse { id, name, input } => {
                        tool_calls.push(openai::ToolCall {
                            id,
                            call_type: "function".to_string(),
                            function: openai::FunctionCall {
                                name,
                                arguments: serde_json::to_string(&input)
                                    .map_err(ProxyError::Serialization)?,
                            },
                        });
                    }
                    anthropic::ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        let text = match content {
                            anthropic::ToolResultContent::Text(s) => s,
                            anthropic::ToolResultContent::Blocks(blocks) => blocks
                                .into_iter()
                                .filter_map(|b| match b {
                                    anthropic::ContentBlock::Text { text, .. } => Some(text),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n"),
                        };
                        result.push(openai::Message {
                            role: "tool".to_string(),
                            content: Some(openai::MessageContent::Text(text)),
                            tool_call_id: Some(tool_use_id),
                            ..Default::default()
                        });
                    }
                    anthropic::ContentBlock::Thinking { thinking } => {
                        if !thinking.is_empty() {
                            reasoning_parts.push(thinking);
                        }
                    }
                }
            }

            if !content_parts.is_empty() || !tool_calls.is_empty() || !reasoning_parts.is_empty() {
                let content = if content_parts.len() == 1 {
                    // A lone text part collapses to a plain string; otherwise keep the parts array.
                    match content_parts.pop().expect("len checked above") {
                        openai::ContentPart::Text { text } => {
                            Some(openai::MessageContent::Text(text))
                        }
                        other => Some(openai::MessageContent::Parts(vec![other])),
                    }
                } else if content_parts.is_empty() {
                    None
                } else {
                    Some(openai::MessageContent::Parts(content_parts))
                };

                result.push(openai::Message {
                    role: msg.role,
                    content,
                    reasoning_content: (!reasoning_parts.is_empty())
                        .then(|| reasoning_parts.join("")),
                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                    ..Default::default()
                });
            }
        }
    }

    Ok(result)
}

pub fn translate_tool(tool: anthropic::Tool) -> openai::Tool {
    openai::Tool {
        tool_type: "function".to_string(),
        function: openai::Function {
            name: tool.name,
            description: tool.description,
            parameters: normalize_schema(tool.input_schema),
        },
    }
}

pub fn is_batch_tool(tool: &anthropic::Tool) -> bool {
    tool.tool_type.as_deref() == Some("BatchTool")
}

pub fn normalize_schema(schema: Value) -> Value {
    match schema {
        Value::Object(mut obj) => {
            obj.retain(|_, value| !value.is_null());

            if obj.get("format").and_then(|v| v.as_str()) == Some("uri") {
                obj.remove("format");
            }

            if let Some(properties) = obj.get_mut("properties").and_then(|v| v.as_object_mut()) {
                for (_, value) in properties.iter_mut() {
                    *value = normalize_schema(std::mem::take(value));
                }
            }

            for key in [
                "items",
                "additionalProperties",
                "contains",
                "not",
                "if",
                "then",
                "else",
            ] {
                if let Some(value) = obj.get_mut(key) {
                    *value = normalize_schema(std::mem::take(value));
                }
            }

            for key in ["allOf", "anyOf", "oneOf", "prefixItems"] {
                if let Some(values) = obj.get_mut(key).and_then(|v| v.as_array_mut()) {
                    for value in values.iter_mut() {
                        *value = normalize_schema(std::mem::take(value));
                    }
                }
            }

            if obj.get("type").and_then(|v| v.as_str()) == Some("object")
                && !obj.contains_key("required")
            {
                obj.insert("required".to_string(), Value::Array(Vec::new()));
            }

            if let Some(required) = obj.get_mut("required") {
                if !required.is_array() {
                    *required = Value::Array(Vec::new());
                }
            }

            Value::Object(obj)
        }
        Value::Array(values) => Value::Array(values.into_iter().map(normalize_schema).collect()),
        other => other,
    }
}

pub fn remove_term(text: &str, term: &str) -> String {
    let tokens: Vec<Vec<u8>> = term
        .split_whitespace()
        .map(|token| {
            token
                .as_bytes()
                .iter()
                .map(u8::to_ascii_lowercase)
                .collect()
        })
        .collect();

    if tokens.is_empty() {
        return text.to_string();
    }

    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut index = 0;

    while index < bytes.len() {
        if let Some(end) = match_term_at(bytes, index, &tokens) {
            spans.push((index, end));
            index = end;
        } else {
            index += 1;
        }
    }

    if spans.is_empty() {
        return text.to_string();
    }

    let mut result = String::with_capacity(text.len());
    let mut cursor = 0;

    for (start, end) in spans {
        result.push_str(&text[cursor..start]);
        cursor = end;
    }

    result.push_str(&text[cursor..]);
    result
}

pub fn map_stop_reason(finish_reason: Option<&str>) -> Option<String> {
    finish_reason.map(|r| {
        match r {
            "tool_calls" => "tool_use",
            "stop" => "end_turn",
            "length" => "max_tokens",
            "content_filter" => "refusal",
            _ => "end_turn",
        }
        .to_string()
    })
}

/// Given the window size, the (true) input token count, and the current `max_tokens`,
/// return a reduced `max_tokens` that leaves the prompt room to fit — or `None` when the
/// prompt alone already fills the window (nothing an output clamp can fix) or no reduction
/// is needed.
///
/// This breaks the deadlock where a conversation is just over the limit: the upstream
/// rejects `input + max_tokens > context`, but `input` alone still fits, so trimming the
/// *output* budget lets the request through (including Claude Code's `/compact`, which
/// otherwise can't run because it too requests output). Headroom is ~2% of the window
/// (min 1024) to absorb any residual tokenizer drift between our count and the upstream's.
pub fn fit_output_to_window(context: u32, input: u32, current_max: Option<u32>) -> Option<u32> {
    const MIN_OUTPUT: u32 = 256;

    let margin = (context / 50).max(1024);
    let available = context.checked_sub(input)?.checked_sub(margin)?;
    let current = current_max.unwrap_or(u32::MAX);

    (available >= MIN_OUTPUT && available < current).then_some(available)
}

/// Parse `(max_context_tokens, input_tokens)` from an OpenAI-style overflow message.
/// The `input_tokens` here is the upstream's *loose lower bound* (`context+1 - max_tokens`),
/// not the true prompt size — callers that can tokenize the request should prefer that.
pub fn parse_context_overflow(message: &str) -> Option<(u32, u32)> {
    let context = leading_number(message.split_once("context length is ")?.1)?;
    let input = message
        .split_once("contains at least ")
        .and_then(|(_, rest)| leading_number(rest))
        .or_else(|| trailing_number(message.split_once(" input tokens")?.0))?;
    Some((context, input))
}

fn leading_number(s: &str) -> Option<u32> {
    s.chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

fn trailing_number(s: &str) -> Option<u32> {
    let reversed: String = s.chars().rev().take_while(|c| c.is_ascii_digit()).collect();
    reversed.chars().rev().collect::<String>().parse().ok()
}

/// Split OpenAI prompt tokens into Anthropic `(input_tokens, cache_read_input_tokens)`.
///
/// OpenAI's `prompt_tokens` is the *total* input including any cache hits, with the
/// cached subset reported under `prompt_tokens_details.cached_tokens`. Anthropic
/// instead reports the non-cached input separately from `cache_read_input_tokens`,
/// so we subtract to avoid double-counting cached tokens in client cost math. The
/// cache figure is `None` when the upstream did not report cached tokens.
pub fn split_prompt_tokens(usage: &openai::Usage) -> (u32, Option<u32>) {
    match usage.prompt_tokens_details.as_ref() {
        Some(details) => (
            usage.prompt_tokens.saturating_sub(details.cached_tokens),
            Some(details.cached_tokens),
        ),
        None => (usage.prompt_tokens, None),
    }
}

/// Translate an Anthropic `tool_choice` into the OpenAI equivalent.
///
/// Returns `(tool_choice, parallel_tool_calls)` where `parallel_tool_calls` carries
/// Anthropic's `disable_parallel_tool_use` (inverted to OpenAI's positive semantics).
pub fn translate_tool_choice(choice: &Value) -> (Option<Value>, Option<bool>) {
    let Some(obj) = choice.as_object() else {
        return (None, None);
    };

    let parallel_tool_calls = obj
        .get("disable_parallel_tool_use")
        .and_then(Value::as_bool)
        .map(|disabled| !disabled);

    let tool_choice = match obj.get("type").and_then(Value::as_str) {
        Some("auto") => Some(Value::from("auto")),
        Some("any") => Some(Value::from("required")),
        Some("none") => Some(Value::from("none")),
        Some("tool") => obj
            .get("name")
            .and_then(Value::as_str)
            .map(|name| serde_json::json!({ "type": "function", "function": { "name": name } })),
        _ => None,
    };

    (tool_choice, parallel_tool_calls)
}

fn match_term_at(text: &[u8], start: usize, tokens: &[Vec<u8>]) -> Option<usize> {
    let mut index = start;

    if is_word_byte(text.get(start).copied())
        && is_word_byte(text.get(start.wrapping_sub(1)).copied())
    {
        return None;
    }

    for (token_index, token) in tokens.iter().enumerate() {
        if token_index > 0 {
            let ws_start = index;
            while index < text.len() && text[index].is_ascii_whitespace() {
                index += 1;
            }
            if ws_start == index {
                return None;
            }
        }

        for expected in token {
            if index >= text.len() || text[index].to_ascii_lowercase() != *expected {
                return None;
            }
            index += 1;
        }
    }

    if is_word_byte(text.get(index.saturating_sub(1)).copied())
        && is_word_byte(text.get(index).copied())
    {
        return None;
    }

    Some(index)
}

fn is_word_byte(byte: Option<u8>) -> bool {
    byte.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn thinking_block_becomes_reasoning_content() {
        let msg = anthropic::Message {
            role: "assistant".to_string(),
            content: anthropic::MessageContent::Blocks(vec![
                anthropic::ContentBlock::Thinking {
                    thinking: "I should preserve this".to_string(),
                },
                anthropic::ContentBlock::Text {
                    text: "Answer".to_string(),
                    cache_control: None,
                },
            ]),
        };

        let translated = translate_message(msg).unwrap();

        assert_eq!(translated.len(), 1);
        assert_eq!(
            translated[0].reasoning_content.as_deref(),
            Some("I should preserve this")
        );
        assert!(matches!(
            translated[0].content,
            Some(openai::MessageContent::Text(_))
        ));
    }

    #[test]
    fn thinking_only_block_still_becomes_assistant_message() {
        let msg = anthropic::Message {
            role: "assistant".to_string(),
            content: anthropic::MessageContent::Blocks(vec![anthropic::ContentBlock::Thinking {
                thinking: "hidden chain".to_string(),
            }]),
        };

        let translated = translate_message(msg).unwrap();

        assert_eq!(translated.len(), 1);
        assert!(translated[0].content.is_none());
        assert_eq!(
            translated[0].reasoning_content.as_deref(),
            Some("hidden chain")
        );
    }

    #[test]
    fn normalize_schema_adds_empty_required_to_object_schemas() {
        let schema = json!({
            "type": "object",
            "properties": {
                "prompt": { "type": "string", "format": "uri" }
            }
        });

        let cleaned = normalize_schema(schema);

        assert_eq!(cleaned["required"], json!([]));
        assert!(cleaned["properties"]["prompt"].get("format").is_none());
    }

    #[test]
    fn normalize_schema_normalizes_non_array_required() {
        let schema = json!({ "type": "object", "required": null });
        let cleaned = normalize_schema(schema);
        assert_eq!(cleaned["required"], json!([]));
    }

    #[test]
    fn normalize_schema_recursively_processes_all_of() {
        let schema = json!({
            "allOf": [
                { "type": "object", "properties": { "a": { "type": "string", "format": "uri" } } },
                { "type": "object", "properties": { "b": { "type": "integer" } } }
            ]
        });

        let cleaned = normalize_schema(schema);

        assert!(cleaned["allOf"][0]["properties"]["a"]
            .get("format")
            .is_none());
        assert_eq!(cleaned["allOf"][0]["required"], json!([]));
        assert_eq!(cleaned["allOf"][1]["required"], json!([]));
    }

    #[test]
    fn normalize_schema_removes_null_values() {
        let schema = json!({
            "type": "object",
            "description": null,
            "properties": { "a": { "type": "string" } }
        });

        let cleaned = normalize_schema(schema);
        assert!(cleaned.get("description").is_none());
    }

    #[test]
    fn remove_term_case_insensitive_with_flexible_whitespace() {
        let result = remove_term("Avoid destructive operations such as RM\t-rF.", "rm -rf");
        assert_eq!(result, "Avoid destructive operations such as .");
    }

    #[test]
    fn remove_term_respects_word_boundaries() {
        let result = remove_term("farm -rf should not match rm -rf", "rm -rf");
        assert_eq!(result, "farm -rf should not match ");
    }

    #[test]
    fn map_stop_reason_translates_all_known_reasons() {
        assert_eq!(map_stop_reason(Some("stop")), Some("end_turn".to_string()));
        assert_eq!(
            map_stop_reason(Some("tool_calls")),
            Some("tool_use".to_string())
        );
        assert_eq!(
            map_stop_reason(Some("length")),
            Some("max_tokens".to_string())
        );
        assert_eq!(
            map_stop_reason(Some("content_filter")),
            Some("refusal".to_string())
        );
        assert_eq!(
            map_stop_reason(Some("unknown")),
            Some("end_turn".to_string())
        );
        assert_eq!(map_stop_reason(None), None);
    }

    #[test]
    fn translate_tool_choice_maps_simple_variants() {
        assert_eq!(
            translate_tool_choice(&json!({"type": "auto"})),
            (Some(json!("auto")), None)
        );
        assert_eq!(
            translate_tool_choice(&json!({"type": "any"})),
            (Some(json!("required")), None)
        );
        assert_eq!(
            translate_tool_choice(&json!({"type": "none"})),
            (Some(json!("none")), None)
        );
    }

    #[test]
    fn translate_tool_choice_maps_specific_tool() {
        let (choice, parallel) = translate_tool_choice(&json!({"type": "tool", "name": "search"}));
        assert_eq!(
            choice,
            Some(json!({"type": "function", "function": {"name": "search"}}))
        );
        assert_eq!(parallel, None);
    }

    #[test]
    fn translate_tool_choice_inverts_disable_parallel() {
        let (choice, parallel) =
            translate_tool_choice(&json!({"type": "any", "disable_parallel_tool_use": true}));
        assert_eq!(choice, Some(json!("required")));
        assert_eq!(parallel, Some(false));
    }

    #[test]
    fn translate_tool_choice_ignores_unknown_shapes() {
        assert_eq!(translate_tool_choice(&json!("auto")), (None, None));
        assert_eq!(
            translate_tool_choice(&json!({"type": "tool"})),
            (None, None)
        );
    }

    #[test]
    fn clamp_reduces_max_tokens_to_fit_context() {
        // The real deadlock: input + 16384 output > 105120. We clamp against the TRUE input
        // (from the tokenizer), here 95011, not the error's loose lower bound.
        // margin = max(105120/50, 1024) = 2102; available = 105120 - 95011 - 2102 = 8007
        let clamped = fit_output_to_window(105120, 95011, Some(16384)).unwrap();
        assert_eq!(clamped, 8007);
        assert!(95011 + clamped <= 105120);
    }

    #[test]
    fn clamp_parses_context_and_input_from_error() {
        let msg = "This model's maximum context length is 105120 tokens. However, you \
                   requested 16384 output tokens and your prompt contains at least 88737 \
                   input tokens, for a total of at least 105121 tokens.";
        assert_eq!(parse_context_overflow(msg), Some((105120, 88737)));
    }

    #[test]
    fn clamp_none_when_prompt_alone_exceeds_window() {
        assert_eq!(fit_output_to_window(1000, 1200, Some(500)), None);
    }

    #[test]
    fn clamp_none_for_unrelated_errors() {
        assert_eq!(parse_context_overflow("invalid api key"), None);
    }

    #[test]
    fn clamp_none_when_available_not_smaller_than_current() {
        // Plenty of room — no need to clamp a small request.
        // available = 105120 - 1000 - 2102 = 102018, which is not < current (8000).
        assert_eq!(fit_output_to_window(105120, 1000, Some(8000)), None);
    }

    #[test]
    fn clamp_parses_input_via_trailing_fallback() {
        // No "contains at least" phrasing — fall back to "<n> input tokens".
        let msg = "maximum context length is 200000 tokens, but you have 100000 input tokens";
        let (context, input) = parse_context_overflow(msg).unwrap();
        assert_eq!((context, input), (200000, 100000));
        // margin = max(200000/50, 1024) = 4000; available = 200000 - 100000 - 4000 = 96000
        assert_eq!(
            fit_output_to_window(context, input, Some(150000)),
            Some(96000)
        );
    }

    #[test]
    fn batch_tool_detected() {
        let tool = anthropic::Tool {
            name: "x".into(),
            description: None,
            input_schema: json!({}),
            tool_type: Some("BatchTool".into()),
        };
        assert!(is_batch_tool(&tool));
    }

    #[test]
    fn regular_tool_not_batch() {
        let tool = anthropic::Tool {
            name: "x".into(),
            description: None,
            input_schema: json!({}),
            tool_type: None,
        };
        assert!(!is_batch_tool(&tool));
    }
}
