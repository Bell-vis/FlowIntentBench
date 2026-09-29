"""Small provider retry policy shared by evaluator and continuation callers.

The policy is transport-only.  It does not decide scientific validity and it
never turns a failed provider call into a scientific pending result.
"""

from __future__ import annotations

from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import hashlib
from typing import Any


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After delta-seconds value or HTTP date."""

    if not value:
        return None
    text = str(value).strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            target = parsedate_to_datetime(text)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            seconds = (target - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds)


def failure_category(error: Any) -> str:
    """Classify a transport failure without importing provider-specific types."""

    status = getattr(error, "http_status", None)
    if status is None:
        status = getattr(error, "code", None)
    if any(code in str(error).casefold() for code in ("insufficient_quota", "billing_hard_limit_reached")):
        return "QUOTA"
    if status == 429 or "http 429" in str(error).casefold() or "rate limit" in str(error).casefold():
        return "RATE_LIMIT"
    if status == 403 and any(
        token in str(error).casefold()
        for token in ("quota", "insufficient", "balance", "billing", "credit")
    ):
        return "QUOTA"
    if status in {408} or "timeout" in str(error).casefold() or "timed out" in str(error).casefold():
        return "TIMEOUT"
    if status in {500, 502, 503, 504}:
        return "UNAVAILABLE"
    return "TRANSPORT_ERROR"


@dataclass(frozen=True)
class ProviderRetryPolicy:
    """Deterministic bounded retry schedule for one provider request."""

    transient_backoff_seconds: float = 2.0
    rate_limit_backoff_seconds: float = 15.0
    rate_limit_max_backoff_seconds: float = 60.0
    jitter_seconds: float = 0.0

    def delay(self, attempt: int, error: Any) -> float:
        if failure_category(error) == "RATE_LIMIT":
            retry_after = getattr(error, "retry_after_seconds", None)
            if retry_after is None:
                retry_after = parse_retry_after(
                    getattr(error, "headers", {}).get("Retry-After")
                    if getattr(error, "headers", None) is not None
                    else None
                )
            delay = (
                float(retry_after)
                if retry_after is not None
                else min(
                    self.rate_limit_backoff_seconds * (2**attempt),
                    self.rate_limit_max_backoff_seconds,
                )
            )
        else:
            delay = self.transient_backoff_seconds * (2**attempt)
        if self.jitter_seconds:
            seed = hashlib.sha256(f"{failure_category(error)}|{attempt}".encode()).digest()
            delay += int.from_bytes(seed[:8], "big") / float(2**64) * self.jitter_seconds
        return max(0.0, float(delay))


__all__ = ["ProviderRetryPolicy", "failure_category", "parse_retry_after"]
