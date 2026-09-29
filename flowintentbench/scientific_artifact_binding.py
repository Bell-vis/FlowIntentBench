"""Semantic dependency bindings for GT and SRAC scientific artifacts.

Ground Truth remains unchanged.  This module stores a lightweight sidecar that
binds existing artifacts to the semantic contract for which they were built.
It also provides fail-closed formal validation and deterministic invalidation
rules.  Presentation identity is tracked separately and never changes the
scientific validity of Ground Truth or an SRAC contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

WORDING_ONLY = "WORDING_ONLY"
FIXED_O_CHANGED = "FIXED_O_CHANGED"
UNRESOLVED_O_SPACE_CHANGED = "UNRESOLVED_O_SPACE_CHANGED"
FINDING_RESPONSIBILITY_CHANGED = "FINDING_RESPONSIBILITY_CHANGED"
SCIENTIFIC_SCOPE_CHANGED = "SCIENTIFIC_SCOPE_CHANGED"
SCIENTIFIC_TARGET_CHANGED = "SCIENTIFIC_TARGET_CHANGED"

_DEPENDENCIES = (
    "presentation",
    "deterministic_g_of_o",
    "ground_truth",
    "scientific_evaluation_contract",
    "srac_response_space",
    "finding_requirement_contract",
)


class StaleScientificArtifactError(ValueError):
    """Raised when formal scientific artifacts do not match case semantics."""

    code = "STALE_SCIENTIFIC_ARTIFACT"


class StalePresentationArtifactError(ValueError):
    """Raised when the presentation binding alone must be regenerated."""

    code = "PRESENTATION_STALE"


def _validate_sha256(value: str, field: str) -> str:
    digest = str(value).strip()
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_value(model_dump(mode="json"))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict())
    if is_dataclass(value):
        return _json_value(asdict(value))
    return value


def canonical_artifact_bytes(value: Any) -> bytes:
    """Return bytes for paths/raw artifacts or canonical bytes for objects."""

    if isinstance(value, Path):
        return value.read_bytes()
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def artifact_sha256(value_or_sha256: Any) -> str:
    """Hash an artifact, or preserve an explicitly supplied SHA-256 digest."""

    if isinstance(value_or_sha256, str) and _SHA256_PATTERN.fullmatch(
        value_or_sha256.strip()
    ):
        return value_or_sha256.strip()
    return hashlib.sha256(canonical_artifact_bytes(value_or_sha256)).hexdigest()


@dataclass(frozen=True)
class ScientificArtifactBinding:
    """Sidecar identity for existing scientific artifacts, not a new GT schema."""

    case_id: str
    semantic_contract_sha256: str
    ground_truth_sha256: str
    scientific_evaluation_contract_sha256: str
    presentation_sha256: str | None = None

    def __post_init__(self) -> None:
        if not str(self.case_id).strip():
            raise ValueError("case_id must be non-empty")
        object.__setattr__(self, "case_id", str(self.case_id).strip())
        for field in (
            "semantic_contract_sha256",
            "ground_truth_sha256",
            "scientific_evaluation_contract_sha256",
        ):
            object.__setattr__(self, field, _validate_sha256(getattr(self, field), field))
        if self.presentation_sha256 is not None:
            object.__setattr__(
                self,
                "presentation_sha256",
                _validate_sha256(self.presentation_sha256, "presentation_sha256"),
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScientificArtifactBinding":
        allowed = {
            "case_id",
            "semantic_contract_sha256",
            "ground_truth_sha256",
            "scientific_evaluation_contract_sha256",
            "presentation_sha256",
        }
        unexpected = set(value) - allowed
        if unexpected:
            raise ValueError(
                "unknown ScientificArtifactBinding fields: "
                + ", ".join(sorted(str(item) for item in unexpected))
            )
        return cls(
            case_id=str(value.get("case_id", "")),
            semantic_contract_sha256=str(
                value.get("semantic_contract_sha256", "")
            ),
            ground_truth_sha256=str(value.get("ground_truth_sha256", "")),
            scientific_evaluation_contract_sha256=str(
                value.get("scientific_evaluation_contract_sha256", "")
            ),
            presentation_sha256=(
                str(value["presentation_sha256"])
                if value.get("presentation_sha256") is not None
                else None
            ),
        )


def _coerce_binding(
    value: ScientificArtifactBinding | Mapping[str, Any],
) -> ScientificArtifactBinding:
    return (
        value
        if isinstance(value, ScientificArtifactBinding)
        else ScientificArtifactBinding.from_mapping(value)
    )


def _contract_source_semantic_sha256(contract: Any) -> str | None:
    if isinstance(contract, Path):
        try:
            contract = json.loads(contract.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    elif isinstance(contract, (bytes, bytearray)):
        try:
            contract = json.loads(bytes(contract).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
    elif isinstance(contract, str) and not _SHA256_PATTERN.fullmatch(
        contract.strip()
    ):
        try:
            contract = json.loads(contract)
        except json.JSONDecodeError:
            return None
    if isinstance(contract, Mapping):
        value = contract.get("source_semantic_contract_sha256")
    else:
        value = getattr(contract, "source_semantic_contract_sha256", None)
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def build_scientific_artifact_binding(
    *,
    case_id: str,
    semantic_contract: Any,
    ground_truth: Any,
    scientific_evaluation_contract: Any,
    presentation: Any | None = None,
) -> ScientificArtifactBinding:
    """Build a sidecar from artifact bytes, objects, paths, or digest strings."""

    return ScientificArtifactBinding(
        case_id=case_id,
        semantic_contract_sha256=artifact_sha256(semantic_contract),
        ground_truth_sha256=artifact_sha256(ground_truth),
        scientific_evaluation_contract_sha256=artifact_sha256(
            scientific_evaluation_contract
        ),
        presentation_sha256=(
            artifact_sha256(presentation) if presentation is not None else None
        ),
    )


def validate_scientific_artifact_binding(
    binding: ScientificArtifactBinding | Mapping[str, Any],
    *,
    case_id: str,
    semantic_contract: Any,
    ground_truth: Any,
    scientific_evaluation_contract: Any,
    presentation: Any | None = None,
    formal_frozen: bool = False,
) -> dict[str, Any]:
    """Validate current artifacts without conflating presentation and science."""

    bound = _coerce_binding(binding)
    semantic_sha256 = artifact_sha256(semantic_contract)
    ground_truth_sha256 = artifact_sha256(ground_truth)
    contract_sha256 = artifact_sha256(scientific_evaluation_contract)
    source_semantic_sha256 = _contract_source_semantic_sha256(
        scientific_evaluation_contract
    )
    contract_supplied_as_digest = (
        isinstance(scientific_evaluation_contract, str)
        and _SHA256_PATTERN.fullmatch(scientific_evaluation_contract.strip())
        is not None
    )
    source_binding_origin = (
        "SCIENTIFIC_EVALUATION_CONTRACT" if source_semantic_sha256 else "NONE"
    )
    scientific_errors: list[str] = []
    presentation_errors: list[str] = []
    warnings: list[str] = []

    if bound.case_id != str(case_id).strip():
        scientific_errors.append("CASE_ID_MISMATCH")
    if bound.semantic_contract_sha256 != semantic_sha256:
        scientific_errors.append("SEMANTIC_CONTRACT_MISMATCH")
    if bound.ground_truth_sha256 != ground_truth_sha256:
        scientific_errors.append("GROUND_TRUTH_MISMATCH")
    if bound.scientific_evaluation_contract_sha256 != contract_sha256:
        scientific_errors.append("SCIENTIFIC_EVALUATION_CONTRACT_MISMATCH")
    if source_semantic_sha256 is None:
        if (
            formal_frozen
            and contract_supplied_as_digest
            and bound.scientific_evaluation_contract_sha256 == contract_sha256
            and bound.semantic_contract_sha256 == semantic_sha256
        ):
            # When the caller intentionally supplies only the frozen SEC
            # digest, the sidecar is the available proof that this exact SEC
            # digest was bound to this exact semantic digest.
            source_semantic_sha256 = bound.semantic_contract_sha256
            source_binding_origin = "SCIENTIFIC_ARTIFACT_BINDING"
        elif formal_frozen:
            scientific_errors.append("SCIENTIFIC_EVALUATION_CONTRACT_UNBOUND")
        else:
            warnings.append("LEGACY_SCIENTIFIC_EVALUATION_CONTRACT_UNBOUND")
    elif not _SHA256_PATTERN.fullmatch(source_semantic_sha256):
        scientific_errors.append("SCIENTIFIC_EVALUATION_CONTRACT_BINDING_INVALID")
    elif source_semantic_sha256 != semantic_sha256:
        scientific_errors.append("SCIENTIFIC_EVALUATION_CONTRACT_SEMANTIC_MISMATCH")

    if presentation is not None:
        presentation_sha256 = artifact_sha256(presentation)
        if bound.presentation_sha256 is None:
            presentation_errors.append("PRESENTATION_BINDING_MISSING")
        elif bound.presentation_sha256 != presentation_sha256:
            presentation_errors.append("PRESENTATION_MISMATCH")
    elif formal_frozen and bound.presentation_sha256 is not None:
        presentation_errors.append("PRESENTATION_ARTIFACT_REQUIRED")

    if scientific_errors:
        status = "STALE_SCIENTIFIC_ARTIFACT"
    elif presentation_errors:
        status = "PRESENTATION_STALE"
    elif warnings:
        status = "LEGACY_COMPATIBLE"
    else:
        status = "VALID"
    return {
        "status": status,
        "scientific_artifact_status": (
            "STALE_SCIENTIFIC_ARTIFACT" if scientific_errors else "VALID"
        ),
        "presentation_status": (
            "PRESENTATION_STALE" if presentation_errors else "VALID"
        ),
        "formal_frozen": formal_frozen,
        "scientific_errors": scientific_errors,
        "presentation_errors": presentation_errors,
        "warnings": warnings,
        "case_id": str(case_id).strip(),
        "semantic_contract_sha256": semantic_sha256,
        "ground_truth_sha256": ground_truth_sha256,
        "scientific_evaluation_contract_sha256": contract_sha256,
        "source_semantic_contract_sha256": source_semantic_sha256,
        "source_semantic_binding_origin": source_binding_origin,
    }


def require_formal_scientific_artifact_binding(
    binding: ScientificArtifactBinding | Mapping[str, Any],
    **artifacts: Any,
) -> dict[str, Any]:
    """Fail closed before release/evaluation when bindings are stale."""

    result = validate_scientific_artifact_binding(
        binding, formal_frozen=True, **artifacts
    )
    if result["scientific_artifact_status"] != "VALID":
        raise StaleScientificArtifactError(
            "STALE_SCIENTIFIC_ARTIFACT: "
            + ", ".join(result["scientific_errors"])
        )
    if result["presentation_status"] != "VALID":
        raise StalePresentationArtifactError(
            "PRESENTATION_STALE: " + ", ".join(result["presentation_errors"])
        )
    return result


def evaluate_scientific_dependency_invalidation(
    change_kind: str,
    *,
    affected_scope_dependencies: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply the frozen semantic dependency invalidation table."""

    kind = str(change_kind).strip().upper()
    statuses = {dependency: "VALID" for dependency in _DEPENDENCIES}
    case_status = "VALID"
    if kind == WORDING_ONLY:
        statuses["presentation"] = "STALE"
    elif kind == FIXED_O_CHANGED:
        statuses.update(
            {
                "presentation": "STALE",
                "deterministic_g_of_o": "STALE",
                "ground_truth": "STALE",
                "scientific_evaluation_contract": "STALE",
            }
        )
        case_status = "STALE"
    elif kind == UNRESOLVED_O_SPACE_CHANGED:
        statuses.update(
            {
                "presentation": "STALE",
                "scientific_evaluation_contract": "STALE",
                "srac_response_space": "STALE",
            }
        )
        case_status = "STALE"
    elif kind == FINDING_RESPONSIBILITY_CHANGED:
        statuses.update(
            {
                "presentation": "STALE",
                "scientific_evaluation_contract": "STALE",
                "finding_requirement_contract": "STALE",
            }
        )
        case_status = "STALE"
    elif kind == SCIENTIFIC_SCOPE_CHANGED:
        affected = (
            tuple(str(item) for item in affected_scope_dependencies)
            if affected_scope_dependencies is not None
            else _DEPENDENCIES
        )
        unknown = sorted(set(affected) - set(_DEPENDENCIES))
        if unknown:
            raise ValueError(
                "unknown affected scope dependencies: " + ", ".join(unknown)
            )
        for dependency in affected:
            statuses[dependency] = "STALE"
        statuses["presentation"] = "STALE"
        case_status = "STALE"
    elif kind == SCIENTIFIC_TARGET_CHANGED:
        statuses = {
            dependency: "NEW_CASE_REQUIRED" for dependency in _DEPENDENCIES
        }
        case_status = "NEW_CASE_REQUIRED"
    else:
        raise ValueError(f"unsupported scientific dependency change: {change_kind}")
    if case_status == "NEW_CASE_REQUIRED":
        artifact_binding_state = "NEW_CASE_REQUIRED"
        scientific_artifact_status = "NEW_CASE_REQUIRED"
    elif case_status == "STALE":
        artifact_binding_state = "STALE_SCIENTIFIC_ARTIFACT"
        scientific_artifact_status = "STALE_SCIENTIFIC_ARTIFACT"
    elif statuses["presentation"] == "STALE":
        artifact_binding_state = "PRESENTATION_STALE"
        scientific_artifact_status = "VALID"
    else:
        artifact_binding_state = "VALID"
        scientific_artifact_status = "VALID"
    presentation_status = (
        "PRESENTATION_STALE" if statuses["presentation"] == "STALE" else "VALID"
    )
    return {
        "change_kind": kind,
        # These identity states describe the effect of the declared semantic or
        # presentation change. Scientific review readiness is intentionally not
        # represented in this module.
        "artifact_binding_state": artifact_binding_state,
        "scientific_artifact_status": scientific_artifact_status,
        "presentation_status": presentation_status,
        "case_reuse_status": case_status,
        "dependencies": statuses,
        "stale_dependencies": sorted(
            key for key, value in statuses.items() if value == "STALE"
        ),
        "new_case_required": case_status == "NEW_CASE_REQUIRED",
    }


__all__ = [
    "FINDING_RESPONSIBILITY_CHANGED",
    "FIXED_O_CHANGED",
    "SCIENTIFIC_SCOPE_CHANGED",
    "SCIENTIFIC_TARGET_CHANGED",
    "UNRESOLVED_O_SPACE_CHANGED",
    "WORDING_ONLY",
    "ScientificArtifactBinding",
    "StalePresentationArtifactError",
    "StaleScientificArtifactError",
    "artifact_sha256",
    "build_scientific_artifact_binding",
    "canonical_artifact_bytes",
    "evaluate_scientific_dependency_invalidation",
    "require_formal_scientific_artifact_binding",
    "validate_scientific_artifact_binding",
]
