"""Scientific portfolio diversity diagnostics and construction targets."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .diversity_audit import build_diversity_report


CONSTRUCTION_TARGETS = {
    "grounded_concept_count": {"minimum": 4, "preferred_maximum": 6},
    "substantive_archetype_count": {"minimum": 4},
    "dataset_count": {"minimum": 3, "preferred_maximum": 4},
    "scientific_entity_type_count": {"minimum": 3},
    "o2_signature_count": {"minimum": 3},
    "dominant_scientific_question_fraction": {"preferred_maximum": 0.5},
}


def build_scientific_diversity_gap_analysis(
    families: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report = build_diversity_report(families, cases)
    entity_defaults = {
        "kitchen_turbulence_activity": "SPATIAL_REGION",
        "kitchen_concentration_heterogeneity": "FIELD",
        "combustor_density_features": "POINT_FEATURE",
        "high_speed_region": "SPATIAL_REGION",
    }
    entity_types = sorted({
        str(item.get("scientific_entity_type") or entity_defaults.get(str(item.get("concept_id")), "SCIENTIFIC_ENTITY"))
        for item in families
    })
    report.update({
        "scientific_entity_types": entity_types,
        "scientific_entity_type_count": len(entity_types),
        "dataset_concept_family_count": int(report.get("family_count", 0)),
        "construction_targets": CONSTRUCTION_TARGETS,
        "target_interpretation": "Targets guide portfolio construction and are not individual family eligibility quotas.",
    })
    return report


def build_dataset_acquisition_requirements() -> list[dict[str, Any]]:
    return [
        {"requirement_id": "vorticity_coherent_structure_dataset", "capabilities": ["velocity vectors", "clear geometry", "reference direction", "vorticity or derivable gradients", "scientific source supporting structure interpretation"]},
        {"requirement_id": "profile_wall_flow_dataset", "capabilities": ["wall geometry", "centerline or cross-section definition", "velocity/pressure", "boundary metadata"]},
        {"requirement_id": "thermal_transport_dataset", "capabilities": ["temperature/enthalpy/species semantics", "geometry", "relevant transport context"]},
        {"requirement_id": "temporal_flow_dataset", "capabilities": ["multiple physical timesteps", "time metadata", "stable coordinate/reference convention"]},
        {"requirement_id": "multiphase_dataset", "capabilities": ["phase variables", "phase semantics", "scientific context", "interfacial observables"]},
    ]


__all__ = ["CONSTRUCTION_TARGETS", "build_dataset_acquisition_requirements", "build_scientific_diversity_gap_analysis"]
