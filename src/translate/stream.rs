use crate::models::anthropic::{
    ContentBlockStart, Delta, DeltaUsage, ErrorData, MessageDeltaData, MessageStartData,
    StreamEvent, Usage,
};
use crate::models::openai;
use crate::translate::core;

#[derive(Debug)]
enum BlockState {
    Idle,
    Thinking { index: usize },
    Text { index: usize },
    ToolUse { index: usize },
}

impl BlockState {
    fn current_index(&self) -> Option<usize> {
        match self {
            Self::Idle => None,
            Self::Thinking { index } | Self::Text { index } | Self::ToolUse { index } => {
                Some(*index)
            }
        }
    }
}

#[derive(Debug)]
pub struct StreamState {
    message_id: Option<String>,
    model: Option<String>,
    fallback_model: String,
    block: BlockState,
    next_index: usize,
    message_started: bool,
}

pub fn initial_state(fallback_model: String) -> StreamState {
    StreamState {
        message_id: None,
        model: None,
        fallback_model,
        block: BlockState::Idle,
        next_index: 0,
        message_started: false,
    }
}

pub fn translate_chunk(state: &mut StreamState, chunk: &openai::StreamChunk) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    if let Some(id) = &chunk.id {
        if state.message_id.is_none() {
            state.message_id = Some(id.clone());
        }
    }
    if let Some(model) = &chunk.model {
        if state.model.is_none() {
            state.model = Some(model.clone());
        }
    }

    let Some(choice) = chunk.choices.first() else {
        return events;
    };

    if !state.message_started {
        events.push(StreamEvent::MessageStart {
            message: MessageStartData {
                id: state
                    .message_id
                    .clone()
                    .unwrap_or_else(|| "msg_proxy".to_string()),
                message_type: "message".to_string(),
                role: "assistant".to_string(),
                model: state
                    .model
                    .clone()
                    .unwrap_or_else(|| state.fallback_model.clone()),
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    ..Default::default()
                },
            },
        });
        state.message_started = true;
    }

    for reasoning in [&choice.delta.reasoning, &choice.delta.reasoning_content]
        .into_iter()
        .flatten()
    {
        emit_reasoning(&mut events, state, reasoning);
    }

    if let Some(content) = &choice.delta.content {
        if !content.is_empty() {
            emit_text(&mut events, state, content);
        }
    }

    if let Some(tool_calls) = &choice.delta.tool_calls {
        emit_tool_calls(&mut events, state, tool_calls);
    }

    if let Some(finish_reason) = &choice.finish_reason {
        emit_finish(&mut events, state, finish_reason, chunk.usage.as_ref());
    }

    events
}

pub fn translate_done(_state: &mut StreamState) -> Vec<StreamEvent> {
    vec![StreamEvent::MessageStop]
}

pub fn translate_error(message: String) -> Vec<StreamEvent> {
    vec![StreamEvent::Error {
        error: ErrorData {
            error_type: "stream_error".to_string(),
            message,
        },
    }]
}

fn close_current_block(events: &mut Vec<StreamEvent>, state: &mut StreamState) {
    if let Some(index) = state.block.current_index() {
        events.push(StreamEvent::ContentBlockStop { index });
        state.next_index = index + 1;
    }
}

fn emit_reasoning(events: &mut Vec<StreamEvent>, state: &mut StreamState, reasoning: &str) {
    if !matches!(state.block, BlockState::Thinking { .. }) {
        close_current_block(events, state);
        let index = state.next_index;
        events.push(StreamEvent::ContentBlockStart {
            index,
            content_block: ContentBlockStart::Thinking {
                thinking: String::new(),
            },
        });
        state.block = BlockState::Thinking { index };
    }

    if let BlockState::Thinking { index } = state.block {
        events.push(StreamEvent::ContentBlockDelta {
            index,
            delta: Delta::ThinkingDelta {
                thinking: reasoning.to_string(),
            },
        });
    }
}

fn emit_text(events: &mut Vec<StreamEvent>, state: &mut StreamState, content: &str) {
    if !matches!(state.block, BlockState::Text { .. }) {
        close_current_block(events, state);
        let index = state.next_index;
        events.push(StreamEvent::ContentBlockStart {
            index,
            content_block: ContentBlockStart::Text {
                text: String::new(),
            },
        });
        state.block = BlockState::Text { index };
    }

    if let BlockState::Text { index } = state.block {
        events.push(StreamEvent::ContentBlockDelta {
            index,
            delta: Delta::TextDelta {
                text: content.to_string(),
            },
        });
    }
}

fn emit_tool_calls(
    events: &mut Vec<StreamEvent>,
    state: &mut StreamState,
    tool_calls: &[openai::DeltaToolCall],
) {
    for tool_call in tool_calls {
        if let Some(id) = &tool_call.id {
            close_current_block(events, state);
            let index = state.next_index;

            if let Some(function) = &tool_call.function {
                if let Some(name) = &function.name {
                    events.push(StreamEvent::ContentBlockStart {
                        index,
                        content_block: ContentBlockStart::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                        },
                    });
                    state.block = BlockState::ToolUse { index };
                }
            }
        }

        if let Some(function) = &tool_call.function {
            if let Some(args) = &function.arguments {
                if let BlockState::ToolUse { index } = state.block {
                    events.push(StreamEvent::ContentBlockDelta {
                        index,
                        delta: Delta::InputJsonDelta {
                            partial_json: args.clone(),
                        },
                    });
                }
            }
        }
    }
}

fn emit_finish(
    events: &mut Vec<StreamEvent>,
    state: &mut StreamState,
    finish_reason: &str,
    usage: Option<&openai::Usage>,
) {
    close_current_block(events, state);

    let stop_reason = core::map_stop_reason(Some(finish_reason));

    let (input_tokens, cache_read_input_tokens) = match usage {
        Some(u) => {
            let (input, cache_read) = core::split_prompt_tokens(u);
            (Some(input), cache_read)
        }
        None => (None, None),
    };

    events.push(StreamEvent::MessageDelta {
        delta: MessageDeltaData {
            stop_reason,
            stop_sequence: None,
        },
        usage: DeltaUsage {
            input_tokens,
            output_tokens: usage.map(|u| u.completion_tokens).unwrap_or(0),
            cache_read_input_tokens,
        },
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn text_chunk(id: &str, model: &str, content: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": { "content": content } }]
        }))
        .unwrap()
    }

    fn reasoning_chunk(id: &str, model: &str, reasoning: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": { "reasoning": reasoning } }]
        }))
        .unwrap()
    }

    fn reasoning_content_chunk(id: &str, model: &str, reasoning: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": { "reasoning_content": reasoning } }]
        }))
        .unwrap()
    }

    fn finish_chunk(id: &str, model: &str, reason: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": reason }]
        }))
        .unwrap()
    }

    fn finish_chunk_with_usage(
        id: &str,
        model: &str,
        reason: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id,
            "model": model,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": reason }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }))
        .unwrap()
    }

    fn tool_start_chunk(id: &str, model: &str, tool_id: &str, name: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": {
                "tool_calls": [{ "index": 0, "id": tool_id, "type": "function",
                    "function": { "name": name } }]
            }}]
        }))
        .unwrap()
    }

    fn tool_args_chunk(id: &str, model: &str, args: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": {
                "tool_calls": [{ "index": 0, "function": { "arguments": args } }]
            }}]
        }))
        .unwrap()
    }

    fn event_types(events: &[StreamEvent]) -> Vec<&str> {
        events.iter().map(|e| e.event_type()).collect()
    }

    #[test]
    fn text_stream_produces_correct_event_sequence() {
        let mut state = initial_state("fallback".into());

        let e1 = translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "Hello"));
        assert_eq!(
            event_types(&e1),
            [
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );

        let e2 = translate_chunk(&mut state, &text_chunk("1", "gpt-4o", " world"));
        assert_eq!(event_types(&e2), ["content_block_delta"]);

        let e3 = translate_chunk(&mut state, &finish_chunk("1", "gpt-4o", "stop"));
        assert_eq!(event_types(&e3), ["content_block_stop", "message_delta"]);

        let e4 = translate_done(&mut state);
        assert_eq!(event_types(&e4), ["message_stop"]);
    }

    #[test]
    fn thinking_then_text_produces_two_blocks() {
        let mut state = initial_state("fallback".into());

        let e1 = translate_chunk(&mut state, &reasoning_chunk("1", "gpt-4o", "Let me think"));
        assert_eq!(
            event_types(&e1),
            [
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );

        let e2 = translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "Answer: 42"));
        assert_eq!(
            event_types(&e2),
            [
                "content_block_stop",
                "content_block_start",
                "content_block_delta"
            ]
        );

        if let StreamEvent::ContentBlockStart { index, .. } = &e2[1] {
            assert_eq!(*index, 1);
        }
    }

    #[test]
    fn reasoning_content_produces_thinking_block() {
        let mut state = initial_state("fallback".into());

        let events = translate_chunk(&mut state, &reasoning_content_chunk("1", "gpt-4o", "Think"));

        assert_eq!(
            event_types(&events),
            [
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );
        if let StreamEvent::ContentBlockDelta { delta, .. } = &events[2] {
            assert!(matches!(delta, Delta::ThinkingDelta { thinking } if thinking == "Think"));
        }
    }

    #[test]
    fn tool_call_stream() {
        let mut state = initial_state("fallback".into());

        let e1 = translate_chunk(
            &mut state,
            &tool_start_chunk("1", "gpt-4o", "call_abc", "read_file"),
        );
        assert_eq!(event_types(&e1), ["message_start", "content_block_start"]);

        if let StreamEvent::ContentBlockStart { content_block, .. } = &e1[1] {
            match content_block {
                ContentBlockStart::ToolUse { id, name } => {
                    assert_eq!(id, "call_abc");
                    assert_eq!(name, "read_file");
                }
                _ => panic!("expected tool_use block"),
            }
        }

        let e2 = translate_chunk(
            &mut state,
            &tool_args_chunk("1", "gpt-4o", "{\"path\":\"/tmp\"}"),
        );
        assert_eq!(event_types(&e2), ["content_block_delta"]);

        let e3 = translate_chunk(&mut state, &finish_chunk("1", "gpt-4o", "tool_calls"));
        assert_eq!(event_types(&e3), ["content_block_stop", "message_delta"]);

        if let StreamEvent::MessageDelta { delta, .. } = &e3[1] {
            assert_eq!(delta.stop_reason.as_deref(), Some("tool_use"));
        }
    }

    #[test]
    fn finish_chunk_maps_cached_tokens_to_cache_read() {
        let mut state = initial_state("fallback".into());
        translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "hi"));

        let chunk: openai::StreamChunk = serde_json::from_value(json!({
            "id": "1", "model": "gpt-4o",
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
            "usage": {
                "prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
                "prompt_tokens_details": { "cached_tokens": 80 }
            }
        }))
        .unwrap();
        let events = translate_chunk(&mut state, &chunk);

        if let StreamEvent::MessageDelta { usage, .. } = &events[1] {
            // input = prompt(100) - cached(80); cache_read = 80
            assert_eq!(usage.input_tokens, Some(20));
            assert_eq!(usage.cache_read_input_tokens, Some(80));
            assert_eq!(usage.output_tokens, 5);
        } else {
            panic!("expected message_delta");
        }
    }

    #[test]
    fn finish_chunk_with_usage_maps_input_and_output_tokens() {
        let mut state = initial_state("fallback".into());

        translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "Hello"));
        let events = translate_chunk(
            &mut state,
            &finish_chunk_with_usage("1", "gpt-4o", "stop", 7, 3),
        );

        if let StreamEvent::MessageDelta { usage, .. } = &events[1] {
            assert_eq!(usage.input_tokens, Some(7));
            assert_eq!(usage.output_tokens, 3);
        } else {
            panic!("expected message_delta");
        }
    }

    #[test]
    fn text_then_tool_call() {
        let mut state = initial_state("fallback".into());

        translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "I'll read that."));

        let e2 = translate_chunk(
            &mut state,
            &tool_start_chunk("1", "gpt-4o", "call_xyz", "read_file"),
        );

        assert!(event_types(&e2).contains(&"content_block_stop"));
        assert!(event_types(&e2).contains(&"content_block_start"));
    }

    #[test]
    fn message_start_uses_chunk_metadata() {
        let mut state = initial_state("my-fallback".into());

        let events = translate_chunk(&mut state, &text_chunk("chatcmpl-42", "gpt-4o", "hi"));

        if let StreamEvent::MessageStart { message } = &events[0] {
            assert_eq!(message.id, "chatcmpl-42");
            assert_eq!(message.model, "gpt-4o");
            assert_eq!(message.role, "assistant");
        }
    }

    #[test]
    fn fallback_model_used_when_chunk_omits_model() {
        let mut state = initial_state("my-fallback".into());

        let chunk: openai::StreamChunk = serde_json::from_value(json!({
            "choices": [{ "index": 0, "delta": { "content": "hey" } }]
        }))
        .unwrap();

        let events = translate_chunk(&mut state, &chunk);

        if let StreamEvent::MessageStart { message } = &events[0] {
            assert_eq!(message.model, "my-fallback");
        }
    }

    #[test]
    fn error_event_produced() {
        let events = translate_error("connection reset".into());
        assert_eq!(event_types(&events), ["error"]);

        if let StreamEvent::Error { error } = &events[0] {
            assert!(error.message.contains("connection reset"));
        }
    }

    #[test]
    fn empty_content_not_emitted() {
        let mut state = initial_state("fallback".into());

        let chunk: openai::StreamChunk = serde_json::from_value(json!({
            "id": "1", "model": "gpt-4o",
            "choices": [{ "index": 0, "delta": { "content": "" } }]
        }))
        .unwrap();

        let events = translate_chunk(&mut state, &chunk);

        let deltas: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ContentBlockDelta { .. }))
            .collect();
        assert!(deltas.is_empty());
    }
}
