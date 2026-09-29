"""Fail-closed classification for external collection failures."""

from __future__ import annotations

from enum import Enum
import re
from typing import Any

from .model_runner import (
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderRequestTimeoutError,
    ProviderUnavailableError,
)


class CollectionStatus(str, Enum):
    BLOCKED_EXTERNAL_QUOTA = "COLLECTION_BLOCKED_EXTERNAL_QUOTA"
    BLOCKED_EXTERNAL_PROVIDER_TIMEOUT = "COLLECTION_BLOCKED_EXTERNAL_PROVIDER_TIMEOUT"
    BLOCKED_EXTERNAL_PROVIDER_UNAVAILABLE = "COLLECTION_BLOCKED_EXTERNAL_PROVIDER_UNAVAILABLE"


def classify_provider_category(error: Any) -> str | None:
    """Return the operational failure category used for reliability reports.

    ``CollectionStatus`` intentionally keeps the historical circuit-breaker
    state for transient provider failures.  The category is more specific:
    HTTP 429 is ``RATE_LIMIT`` (provider capacity pressure), whereas quota
    failures are ``QUOTA`` and transport/read failures are ``UNAVAILABLE`` or
    ``TIMEOUT``.  This preserves compatibility for existing callers while
    preventing 429 from being described as an Internet outage.
    """

    text = str(error or "").casefold()
    if any(code in text for code in ("insufficient_quota", "billing_hard_limit_reached")):
        return "QUOTA"
    if isinstance(error, ProviderRateLimitError) or "429" in text or "rate limit" in text:
        return "RATE_LIMIT"
    if isinstance(error, ProviderQuotaError) or (re.search(r"\b403\b", text) and any(
        token in text for token in ("quota", "insufficient", "balance", "billing", "credit")
    )):
        return "QUOTA"
    if any(token in text for token in ("quota", "insufficient_quota", "resource exhausted")):
        return "QUOTA"
    if isinstance(error, ProviderRequestTimeoutError):
        return "TIMEOUT"
    if isinstance(error, ProviderUnavailableError):
        return "UNAVAILABLE"
    if "timeout" in text or "timed out" in text:
        return "TIMEOUT"
    if any(
        token in text
        for token in (
            "provider unavailable",
            "unexpected_eof",
            "ssl:",
            "connection reset",
            "remote end closed",
            "connection refused",
            "transport failed after",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
        )
    ):
        return "UNAVAILABLE"
    return None


def classify_provider_failure(error: Any) -> CollectionStatus | None:
    """Classify only known external provider failures, never runtime/scientific errors."""

    text = str(error or "").casefold()
    if any(code in text for code in ("insufficient_quota", "billing_hard_limit_reached")):
        return CollectionStatus.BLOCKED_EXTERNAL_QUOTA
    # A transient HTTP 429/rate-limit response is provider pressure, not an
    # account quota exhaustion.  It participates in the circuit breaker and
    # may be retried.  Account/quota failures (notably HTTP 403 with an
    # insufficient_*_quota code) remain an immediate external-quota block.
    if isinstance(error, ProviderRateLimitError) or "429" in text or "rate limit" in text:
        return CollectionStatus.BLOCKED_EXTERNAL_PROVIDER_UNAVAILABLE
    if isinstance(error, ProviderQuotaError) or (re.search(r"\b403\b", text) and any(
        token in text for token in ("quota", "insufficient", "balance", "billing", "credit")
    )):
        return CollectionStatus.BLOCKED_EXTERNAL_QUOTA
    if any(token in text for token in ("quota", "insufficient_quota", "resource exhausted")):
        return CollectionStatus.BLOCKED_EXTERNAL_QUOTA
    if isinstance(error, ProviderRequestTimeoutError):
        return CollectionStatus.BLOCKED_EXTERNAL_PROVIDER_TIMEOUT
    if isinstance(error, ProviderUnavailableError):
        return CollectionStatus.BLOCKED_EXTERNAL_PROVIDER_UNAVAILABLE
    # Retry exhaustion wraps the original transport exception as text before
    # it reaches collection-state accounting. Preserve provider-unavailable
    # semantics for common TLS/socket and 5xx failures so the case-level
    # circuit breaker can pause after repeated gateway outages.
    if any(
        token in text
        for token in (
            "provider unavailable",
            "unexpected_eof",
            "ssl:",
            "connection reset",
            "remote end closed",
            "connection refused",
            "transport failed after",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
        )
    ):
        return CollectionStatus.BLOCKED_EXTERNAL_PROVIDER_UNAVAILABLE
    if "timeout" in text or "timed out" in text:
        return CollectionStatus.BLOCKED_EXTERNAL_PROVIDER_TIMEOUT
    return None


def provider_failure_info(error: Any, *, fallback_status: CollectionStatus | str | None = None) -> dict[str, Any] | None:
    """Return structured provider failure metadata without treating strings as types."""

    status = classify_provider_failure(error)
    if status is None and fallback_status is not None:
        fallback_value = fallback_status.value if isinstance(fallback_status, CollectionStatus) else str(fallback_status)
        status = CollectionStatus(fallback_value) if fallback_value in {item.value for item in CollectionStatus} else None
    if status is None:
        return None
    text = str(error or "")
    match = re.search(r"\b(4\d\d|5\d\d)\b", text)
    category = classify_provider_category(error) or {
        CollectionStatus.BLOCKED_EXTERNAL_QUOTA: "QUOTA",
        CollectionStatus.BLOCKED_EXTERNAL_PROVIDER_TIMEOUT: "TIMEOUT",
        CollectionStatus.BLOCKED_EXTERNAL_PROVIDER_UNAVAILABLE: "UNAVAILABLE",
    }[status]
    if isinstance(error, ProviderRequestTimeoutError):
        exception_type = type(error).__name__
    elif isinstance(error, ProviderUnavailableError):
        exception_type = type(error).__name__
    elif category == "QUOTA":
        exception_type = "ProviderQuotaError"
    elif category == "RATE_LIMIT":
        exception_type = "ProviderRateLimitError"
    else:
        exception_type = "ProviderTransportError"
    return {
        "category": category,
        "collection_status": status.value,
        "exception_type": exception_type,
        "http_status": int(match.group(1)) if match else None,
        "provider_code": None,
        "message": text or status.value,
        "retry_after_seconds": getattr(error, "retry_after_seconds", None),
    }


__all__ = [
    "CollectionStatus",
    "classify_provider_category",
    "classify_provider_failure",
    "provider_failure_info",
]
