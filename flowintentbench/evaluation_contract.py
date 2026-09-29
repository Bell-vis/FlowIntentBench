"""Human-confirmation boundary for compiled scientific evaluation contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .srac import ScientificEvaluationContract


@dataclass(frozen=True)
class FrozenEvaluationCase:
    """Validated adapter for the artifacts consumed by formal evaluation.

    This is intentionally an adapter, not a second case or GT schema.  It
    keeps the existing model-facing input, construction metadata, GT, SEC and
    scientific-binding sidecars together and performs the same hash/binding
    checks used by release validation before a formal evaluator is built.
    """

    case_input: Any
    case_metadata: Any
    ground_truth: Any
    scientific_evaluation_contract: ScientificEvaluationContract
    scientific_artifact_binding: Any
    contract_confirmation: Mapping[str, Any] | None = None
    case_dir: Path | None = None
    formal_frozen: bool = False

    @classmethod
    def from_paths(
        cls,
        case_dir: str | Path,
        *,
        formal_frozen: bool = False,
    ) -> "FrozenEvaluationCase":
        root = Path(case_dir)
        def read(name: str) -> Any:
            try:
                return json.loads((root / name).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"failed to load evaluation artifact: {root / name}") from exc

        from .case_design import CaseConstructionMetadata
        from .ground_truth import load_ground_truth
        from .schema import validate_model_input
        from .scientific_artifact_binding import (
            ScientificArtifactBinding,
            require_formal_scientific_artifact_binding,
            validate_scientific_artifact_binding,
        )

        raw_input = read("case_input.json")
        case_input = validate_model_input(raw_input)
        metadata = CaseConstructionMetadata.model_validate(read("case_construction_metadata.json"))
        evidence_payload: Any = []
        evidence_path = root / "agent_ready_evidence.json"
        if evidence_path.is_file():
            evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence = evidence_payload if isinstance(evidence_payload, list) else []
        ground_truth = load_ground_truth(root / "ground_truth.json", case_metadata=metadata, evidence_records=evidence)
        sec_raw = read("scientific_evaluation_contract.json")
        if not isinstance(sec_raw, Mapping):
            raise ValueError("scientific_evaluation_contract.json must be an object")
        sec = ScientificEvaluationContract(
            case_id=str(sec_raw.get("case_id", "")),
            enumerated_valid_O=tuple(sec_raw.get("enumerated_valid_O", ()) or ()),
            explicitly_invalid_O=tuple(sec_raw.get("explicitly_invalid_O", ()) or ()),
            unenumerated_policy=str(sec_raw.get("unenumerated_policy", "")),
            G_of_O=tuple(sec_raw.get("G_of_O", ()) or ()),
            finding_requirement_contract=sec_raw.get("finding_requirement_contract", {}),
            adequate_core_sets=tuple(tuple(item) for item in sec_raw.get("adequate_core_sets", ()) or ()),
            tolerances=sec_raw.get("tolerances", {}),
            adjudication_provenance=tuple(sec_raw.get("adjudication_provenance", ()) or ()),
            status=str(sec_raw.get("status", "DRAFT_REQUIRES_HUMAN_CONFIRMATION")),
            adjudicated_findings=tuple(sec_raw.get("adjudicated_findings", ()) or ()),
            source_semantic_contract_sha256=sec_raw.get("source_semantic_contract_sha256"),
        )
        binding = ScientificArtifactBinding.from_mapping(read("scientific_artifact_binding.json"))
        semantic = read("scientific_semantic_projection.json")
        presentation_payload = read("scientific_question.json")
        presentation = (
            presentation_payload.get("scientific_question")
            if isinstance(presentation_payload, Mapping)
            else presentation_payload
        )
        binding_kwargs = {
            "binding": binding,
            "case_id": metadata.case_id,
            "semantic_contract": semantic,
            "ground_truth": ground_truth,
            "scientific_evaluation_contract": sec,
            "presentation": presentation,
        }
        confirmation_path = root / "scientific_evaluation_contract_confirmation.json"
        confirmation = read(confirmation_path.name) if confirmation_path.is_file() else None
        if formal_frozen:
            require_formal_scientific_artifact_binding(**binding_kwargs)
            if not isinstance(confirmation, Mapping) or confirmation.get("status") != "CONFIRMED":
                raise ValueError("FORMAL_EVALUATION requires confirmed SEC review artifact")
            confirmation_result = validate_contract_confirmation(confirmation)
            if confirmation_result["status"] != "PASS":
                raise ValueError("SEC confirmation artifact is invalid")
            from .scientific_artifact_binding import artifact_sha256
            if confirmation.get("scientific_evaluation_contract_sha256") != artifact_sha256(sec):
                raise ValueError("SEC confirmation digest does not match frozen SEC")
        else:
            result = validate_scientific_artifact_binding(**binding_kwargs, formal_frozen=False)
            if result["scientific_artifact_status"] not in {"VALID", "VALID_WITH_WARNINGS"}:
                raise ValueError("scientific artifact binding is stale")
        if sec.case_id != metadata.case_id:
            raise ValueError("SEC case_id does not match construction metadata")
        return cls(case_input, metadata, ground_truth, sec, binding, confirmation, root, formal_frozen)

    def validate(self) -> dict[str, Any]:
        """Return a compact status without changing any upstream artifact."""

        return {
            "status": "PASS",
            "case_id": self.case_metadata.case_id,
            "formal_frozen": self.formal_frozen,
            "scientific_evaluation_contract_status": self.scientific_evaluation_contract.status,
        }


def validate_contract_confirmation(value: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if str(value.get("status", "")) == "CONFIRMED" and not value.get("human_review_artifact"):
        errors.append("contract confirmation requires human-review artifact")
    if any(item.get("rationale") in {None, ""} for item in value.get("explicitly_invalid_O", []) if isinstance(item, Mapping)):
        errors.append("explicitly invalid O requires rationale")
    digest = value.get("scientific_evaluation_contract_sha256")
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
        errors.append("scientific_evaluation_contract_sha256 must be a SHA-256 digest")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors, "human_confirmation_required": True}


__all__ = ["FrozenEvaluationCase", "ScientificEvaluationContract", "validate_contract_confirmation"]
