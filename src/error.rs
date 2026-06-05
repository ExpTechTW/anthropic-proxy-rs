use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

/// Application-specific errors
#[derive(Error, Debug)]
pub enum ProxyError {
    #[error("Request transformation error: {0}")]
    Transform(String),

    /// A transport/connection failure with no usable HTTP status (mapped to 502).
    #[error("Upstream API error: {0}")]
    Upstream(String),

    /// An upstream HTTP error whose status code should be surfaced to the client
    /// as-is (e.g. a 400 bad request or 429 rate limit) rather than masked as 502.
    #[error("Upstream API error ({status}): {message}")]
    UpstreamStatus { status: StatusCode, message: String },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}

/// Map an HTTP status to the matching Anthropic error `type` string so clients
/// (Claude Code, the official SDKs) classify failures correctly.
fn anthropic_error_type(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        529 => "overloaded_error",
        _ => "api_error",
    }
}

impl ProxyError {
    /// The HTTP status this error maps to.
    pub fn status_code(&self) -> StatusCode {
        match self {
            ProxyError::Transform(_) | ProxyError::Serialization(_) => StatusCode::BAD_REQUEST,
            ProxyError::Upstream(_) | ProxyError::Http(_) => StatusCode::BAD_GATEWAY,
            ProxyError::UpstreamStatus { status, .. } => *status,
        }
    }

    fn message(self) -> String {
        match self {
            ProxyError::Transform(msg) | ProxyError::Upstream(msg) => msg,
            ProxyError::UpstreamStatus { message, .. } => message,
            ProxyError::Serialization(err) => format!("JSON error: {}", err),
            ProxyError::Http(err) => format!("HTTP error: {}", err),
        }
    }
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_type = anthropic_error_type(status);
        let body = Json(json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": self.message(),
            }
        }));

        (status, body).into_response()
    }
}

/// Result type for proxy operations
pub type ProxyResult<T> = Result<T, ProxyError>;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    fn status_of(error: ProxyError) -> StatusCode {
        error.into_response().status()
    }

    #[test]
    fn transform_error_returns_400() {
        assert_eq!(
            status_of(ProxyError::Transform("bad".into())),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn upstream_error_returns_502() {
        assert_eq!(
            status_of(ProxyError::Upstream("bad".into())),
            StatusCode::BAD_GATEWAY
        );
    }

    #[test]
    fn serialization_error_returns_400() {
        let err: serde_json::Error = serde_json::from_str::<String>("not json").unwrap_err();
        assert_eq!(
            status_of(ProxyError::Serialization(err)),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn upstream_status_preserves_code() {
        for code in [400u16, 401, 403, 404, 413, 429] {
            let status = StatusCode::from_u16(code).unwrap();
            let err = ProxyError::UpstreamStatus {
                status,
                message: "boom".into(),
            };
            assert_eq!(status_of(err), status);
        }
    }

    #[tokio::test]
    async fn error_body_uses_anthropic_envelope() {
        use axum::body::to_bytes;

        let err = ProxyError::UpstreamStatus {
            status: StatusCode::BAD_REQUEST,
            message: "messages: at least one message is required".into(),
        };
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(
            json["error"]["message"],
            "messages: at least one message is required"
        );
    }

    #[test]
    fn status_code_accessor_matches_response() {
        let err = ProxyError::Transform("bad".into());
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
    }
}
