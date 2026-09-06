"""Provider diagnostics without exception messages, headers, keys or prompts."""
import logging

def generation_diagnostic(exc, model):
    status = getattr(exc, "status_code", None)
    status = status if type(status) is int and 100 <= status <= 599 else None
    kind = type(exc).__name__
    if kind not in {"RateLimitError", "AuthenticationError", "PermissionDeniedError",
                    "NotFoundError", "BadRequestError", "InternalServerError",
                    "APITimeoutError", "APIConnectionError", "APIStatusError"}:
        kind = "UnexpectedError"
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    if not isinstance(code, str) or code not in {
        "rate_limit_exceeded", "model_not_found", "model_decommissioned",
        "model_permission_blocked", "invalid_api_key", "context_length_exceeded",
        "tokens_limit_reached", "insufficient_quota",
    }:
        code = "unknown"
    if model not in {"llama-3.3-70b-versatile", "openai/gpt-oss-120b"}:
        model = "other"
    logging.warning("Generation failed: model=%s type=%s status=%s code=%s",
                    model, kind, status, code)
    if status == 429:
        reason = "The provider returned a usage limit (HTTP 429)."
    elif status == 401:
        reason = "The provider rejected authentication (HTTP 401)."
    elif status == 403:
        reason = "The provider denied access (HTTP 403)."
    elif status == 404 or code in {"model_not_found", "model_decommissioned"}:
        reason = "The requested model is unavailable."
    elif kind == "APITimeoutError":
        reason = "The answer request timed out."
    elif kind == "APIConnectionError":
        reason = "The answer service could not be reached."
    elif status == 400:
        reason = "The provider rejected this request (HTTP 400)."
    elif status is not None and status >= 500:
        reason = "The answer provider reported a server error."
    else:
        reason = "The answer could not be generated."
    return reason + " Your retrieved reviews remain available. Try another model or retry later."
