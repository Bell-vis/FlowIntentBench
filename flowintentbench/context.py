"""Internal evidence-backed construction of dataset and case context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal, Mapping, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from .schema import (
    BoundaryContext,
    CaseContext,
    GeometryAsset,
    ObjectContext,
    OperatingConditionContext,
    ReferenceDirectionContext,
    RegionContext,
)


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SourceType = Literal[
    "original_publication",
    "direct_analysis_publication",
    "dataset_documentation",
    "example_source_code",
    "relevant_scientific_reference",
]
EvidenceType = Literal["context", "operationalization", "finding"]
ReviewStatus = Literal["pending", "approved", "rejected"]
ContextField = Literal[
    "physical_setting",
    "objects",
    "boundaries",
    "regions",
    "reference_directions",
    "operating_conditions",
    "geometry_assets",
]

_CONTEXT_FIELD_ORDER = {
    "physical_setting": 0,
    "objects": 1,
    "boundaries": 2,
    "regions": 3,
    "reference_directions": 4,
    "operating_conditions": 5,
    "geometry_assets": 6,
}
_NAMED_CONTEXT_FIELDS = {
    "objects",
    "boundaries",
    "regions",
    "reference_directions",
    "operating_conditions",
}


class ContextConstructionError(ValueError):
    """Raised when internal context resources cannot be built consistently."""


class ContextConstructionModel(BaseModel):
    """Closed schema for benchmark-internal context resources."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SourceRecord(ContextConstructionModel):
    source_id: NonEmptyString
    source_type: SourceType
    title: NonEmptyString
    authors: list[NonEmptyString] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=0)
    venue: NonEmptyString | None = None
    doi: NonEmptyString | None = None
    report_id: NonEmptyString | None = None


def _validate_unique_source_ids(sources: Iterable[SourceRecord]) -> None:
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source_id values must be unique within a dataset")


class SourceCollection(ContextConstructionModel):
    """Dataset-scoped collection of accepted, traceable sources.

    An empty collection is valid during construction. Formal benchmark release
    validation is requested through
    ``validate_source_collection(..., require_formal_source=True)``.
    """

    dataset_id: NonEmptyString
    sources: list[SourceRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_ids(self) -> "SourceCollection":
        _validate_unique_source_ids(self.sources)
        return self

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize without changing source order or source IDs."""

        validated = validate_source_collection(self)
        return _dump_source_collection_json(validated, indent=indent)

    def write_json(self, path: str | Path, *, indent: int = 2) -> "SourceCollection":
        """Write a validated source collection to the given JSON file path."""

        validated = validate_source_collection(self)
        Path(path).write_text(
            _dump_source_collection_json(validated, indent=indent),
            encoding="utf-8",
        )
        return validated

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        dataset_id: str | None = None,
        require_formal_source: bool = False,
    ) -> "SourceCollection":
        return load_source_collection(
            path,
            dataset_id=dataset_id,
            require_formal_source=require_formal_source,
        )


def _dump_source_collection_json(collection: SourceCollection, *, indent: int = 2) -> str:
    return json.dumps(
        collection.model_dump(mode="json"),
        ensure_ascii=False,
        indent=indent,
    ) + "\n"


class EvidenceLocator(ContextConstructionModel):
    section: str | None = None
    page: str | int | None = None
    figure: str | int | None = None
    code_location: str | None = None


class _ContextFactBase(ContextConstructionModel):
    fact_id: NonEmptyString
    review_status: ReviewStatus


class PhysicalSettingFact(_ContextFactBase):
    field: Literal["physical_setting"]
    value: NonEmptyString


class ObjectFact(_ContextFactBase):
    field: Literal["objects"]
    value: ObjectContext


class BoundaryFact(_ContextFactBase):
    field: Literal["boundaries"]
    value: BoundaryContext


class RegionFact(_ContextFactBase):
    field: Literal["regions"]
    value: RegionContext


class ReferenceDirectionFact(_ContextFactBase):
    field: Literal["reference_directions"]
    value: ReferenceDirectionContext


class OperatingConditionFact(_ContextFactBase):
    field: Literal["operating_conditions"]
    value: OperatingConditionContext


class GeometryAssetFact(_ContextFactBase):
    field: Literal["geometry_assets"]
    value: GeometryAsset


ContextFact: TypeAlias = Annotated[
    PhysicalSettingFact
    | ObjectFact
    | BoundaryFact
    | RegionFact
    | ReferenceDirectionFact
    | OperatingConditionFact
    | GeometryAssetFact,
    Field(discriminator="field"),
]


class EvidenceRecord(ContextConstructionModel):
    evidence_id: NonEmptyString
    dataset_id: NonEmptyString
    evidence_type: EvidenceType
    statement: NonEmptyString
    source_id: NonEmptyString
    locator: EvidenceLocator = Field(default_factory=EvidenceLocator)
    eligible_for_context: bool
    context_facts: list[ContextFact] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_boundary(self) -> "EvidenceRecord":
        if self.evidence_type != "context":
            if self.eligible_for_context:
                raise ValueError(
                    "operationalization and finding evidence cannot be eligible for context"
                )
            if self.context_facts:
                raise ValueError(
                    "operationalization and finding evidence must not contain context_facts"
                )
        fact_ids = [fact.fact_id for fact in self.context_facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("context_facts fact_id values must be unique within an evidence record")
        return self


class DatasetContext(CaseContext):
    """Dataset-level approved context pool using the model-facing value schema."""


class FactLocation(ContextConstructionModel):
    field: ContextField
    position: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_position(self) -> "FactLocation":
        if self.field == "physical_setting" and self.position is not None:
            raise ValueError("physical_setting fact location must have position=null")
        if self.field != "physical_setting" and self.position is None:
            raise ValueError("array context fact locations require a position")
        return self


class _FactProvenance(ContextConstructionModel):
    dataset_id: NonEmptyString
    fact_index: dict[str, FactLocation]
    fact_provenance: dict[str, list[NonEmptyString]]

    @field_validator("fact_index")
    @classmethod
    def validate_fact_index(cls, value: dict[str, FactLocation]) -> dict[str, FactLocation]:
        if any(not fact_id.strip() for fact_id in value):
            raise ValueError("fact_index keys must be non-empty")
        locations = [(item.field, item.position) for item in value.values()]
        if len(locations) != len(set(locations)):
            raise ValueError("fact_index locations must be unique")
        return value

    @field_validator("fact_provenance")
    @classmethod
    def validate_fact_provenance(
        cls, value: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        for fact_id, evidence_ids in value.items():
            if not fact_id.strip():
                raise ValueError("fact_provenance keys must be non-empty")
            if not evidence_ids:
                raise ValueError("every fact must have at least one supporting evidence ID")
            if len(evidence_ids) != len(set(evidence_ids)):
                raise ValueError("supporting evidence IDs must be unique")
        return value

    @model_validator(mode="after")
    def validate_fact_sets(self) -> "_FactProvenance":
        if set(self.fact_index) != set(self.fact_provenance):
            raise ValueError("fact_index and fact_provenance must contain the same fact IDs")
        return self


class DatasetContextProvenance(_FactProvenance):
    pass


class CaseContextProvenance(_FactProvenance):
    case_id: NonEmptyString


class ContextSelection(ContextConstructionModel):
    case_id: NonEmptyString
    dataset_id: NonEmptyString
    include_fact_ids: list[NonEmptyString] = Field(default_factory=list)

    @field_validator("include_fact_ids")
    @classmethod
    def validate_include_fact_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("include_fact_ids must be unique")
        return value


@dataclass(frozen=True)
class DatasetContextBuildResult:
    context: DatasetContext
    provenance: DatasetContextProvenance


@dataclass(frozen=True)
class CaseContextSelectionResult:
    context: CaseContext
    provenance: CaseContextProvenance


def _validated_value(fact: ContextFact) -> Any:
    if isinstance(fact.value, BaseModel):
        return fact.value.model_dump(mode="json")
    return fact.value


def _canonical_payload(fact: ContextFact) -> str:
    return json.dumps(
        {"field": fact.field, "value": _validated_value(fact)},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _context_locations(context: CaseContext) -> set[tuple[ContextField, int | None]]:
    locations: set[tuple[ContextField, int | None]] = set()
    if context.physical_setting is not None:
        locations.add(("physical_setting", None))
    for field in (
        "objects",
        "boundaries",
        "regions",
        "reference_directions",
        "operating_conditions",
        "geometry_assets",
    ):
        locations.update((field, position) for position, _ in enumerate(getattr(context, field)))
    return locations


def _coerce_sources(
    records: SourceCollection | Iterable[SourceRecord | Mapping[str, Any]],
) -> tuple[SourceRecord, ...]:
    if isinstance(records, SourceCollection):
        return tuple(records.sources)
    return tuple(
        item if isinstance(item, SourceRecord) else SourceRecord.model_validate(item)
        for item in records
    )


def _coerce_evidence(
    records: Iterable[EvidenceRecord | Mapping[str, Any]],
) -> tuple[EvidenceRecord, ...]:
    return tuple(
        item if isinstance(item, EvidenceRecord) else EvidenceRecord.model_validate(item)
        for item in records
    )


class SourceRecordValidator:
    """Validate source identity fields according to the source's scientific role."""

    _PUBLICATION_LIKE_TYPES = {
        "original_publication",
        "direct_analysis_publication",
        "relevant_scientific_reference",
    }

    @classmethod
    def validate(
        cls,
        source: SourceRecord | Mapping[str, Any],
    ) -> SourceRecord:
        record = source if isinstance(source, SourceRecord) else SourceRecord.model_validate(source)

        has_complete_publication_identity = (
            bool(record.authors)
            and record.year is not None
            and record.venue is not None
        )
        if record.source_type in cls._PUBLICATION_LIKE_TYPES:
            if not (record.doi or record.report_id or has_complete_publication_identity):
                raise ContextConstructionError(
                    f"source {record.source_id!r} requires doi, report_id, or "
                    "title + authors + year + venue"
                )
        elif record.venue is None:
            raise ContextConstructionError(
                f"source {record.source_id!r} requires venue for {record.source_type}"
            )
        return record


class SourceIdUniquenessValidator:
    """Validate source records and enforce dataset-local source_id uniqueness."""

    @staticmethod
    def validate(
        sources: SourceCollection | Iterable[SourceRecord | Mapping[str, Any]],
    ) -> tuple[SourceRecord, ...]:
        validated = tuple(SourceRecordValidator.validate(source) for source in _coerce_sources(sources))
        try:
            _validate_unique_source_ids(validated)
        except ValueError as exc:
            raise ContextConstructionError(str(exc)) from exc
        return validated


def validate_source_collection(
    collection: SourceCollection | Mapping[str, Any],
    *,
    expected_dataset_id: str | None = None,
    require_formal_source: bool = False,
) -> SourceCollection:
    """Validate a dataset-scoped source collection without changing IDs or order."""

    validated = (
        collection
        if isinstance(collection, SourceCollection)
        else SourceCollection.model_validate(collection)
    )
    if expected_dataset_id is not None and validated.dataset_id != expected_dataset_id:
        raise ContextConstructionError(
            f"source collection belongs to dataset {validated.dataset_id!r}, "
            f"expected {expected_dataset_id!r}"
        )
    sources = SourceIdUniquenessValidator.validate(validated)
    has_dataset_traceable_source = any(
        source.source_type != "relevant_scientific_reference" for source in sources
    )
    if require_formal_source and not has_dataset_traceable_source:
        raise ContextConstructionError(
            "formal benchmark datasets require at least one accepted, dataset-traceable source"
        )
    if tuple(validated.sources) != sources:
        return SourceCollection(dataset_id=validated.dataset_id, sources=list(sources))
    return validated


def load_source_collection(
    path: str | Path,
    *,
    dataset_id: str | None = None,
    require_formal_source: bool = False,
) -> SourceCollection:
    """Load and validate a ``sources.json`` file."""

    source_path = Path(path)
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextConstructionError(f"failed to load sources.json: {source_path}") from exc
    return validate_source_collection(
        payload,
        expected_dataset_id=dataset_id,
        require_formal_source=require_formal_source,
    )


def save_source_collection(
    collection: SourceCollection | Mapping[str, Any],
    path: str | Path,
    *,
    expected_dataset_id: str | None = None,
    require_formal_source: bool = False,
    indent: int = 2,
) -> SourceCollection:
    """Validate and write a ``sources.json`` file."""

    validated = validate_source_collection(
        collection,
        expected_dataset_id=expected_dataset_id,
        require_formal_source=require_formal_source,
    )
    Path(path).write_text(
        _dump_source_collection_json(validated, indent=indent),
        encoding="utf-8",
    )
    return validated


class EvidenceSourceCrossReferenceValidator:
    """Validate dataset-scoped source/evidence IDs and source-role semantics."""

    @staticmethod
    def validate(
        dataset_id: str,
        sources: SourceCollection | Iterable[SourceRecord | Mapping[str, Any]],
        evidence_records: Iterable[EvidenceRecord | Mapping[str, Any]],
    ) -> tuple[tuple[SourceRecord, ...], tuple[EvidenceRecord, ...]]:
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ContextConstructionError("dataset_id must be a non-empty string")
        if isinstance(sources, SourceCollection) and sources.dataset_id != dataset_id:
            raise ContextConstructionError(
                f"source collection belongs to dataset {sources.dataset_id!r}, "
                f"expected {dataset_id!r}"
            )
        validated_sources = SourceIdUniquenessValidator.validate(sources)
        validated_evidence = _coerce_evidence(evidence_records)

        evidence_ids = [item.evidence_id for item in validated_evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ContextConstructionError("evidence_id values must be unique within a dataset")

        known_sources = {source.source_id: source for source in validated_sources}
        for evidence in validated_evidence:
            if evidence.dataset_id != dataset_id:
                raise ContextConstructionError(
                    f"evidence {evidence.evidence_id!r} belongs to dataset "
                    f"{evidence.dataset_id!r}, expected {dataset_id!r}"
                )
            if evidence.source_id not in known_sources:
                raise ContextConstructionError(
                    f"evidence {evidence.evidence_id!r} references unknown source "
                    f"{evidence.source_id!r}"
                )
            source = known_sources[evidence.source_id]
            if (
                source.source_type == "relevant_scientific_reference"
                and evidence.evidence_type != "operationalization"
            ):
                raise ContextConstructionError(
                    "relevant_scientific_reference may only support "
                    "operationalization evidence"
                )
        return validated_sources, validated_evidence


class ContextSchemaValidator:
    """Validate context values and their case-level file separation."""

    @staticmethod
    def validate(
        context: CaseContext | Mapping[str, Any],
        *,
        data_file_paths: Iterable[str] = (),
    ) -> CaseContext:
        validated = context if isinstance(context, CaseContext) else CaseContext.model_validate(context)
        overlap = {item.path for item in validated.geometry_assets} & set(data_file_paths)
        if overlap:
            raise ContextConstructionError(
                f"geometry assets must not also be data files: {sorted(overlap)}"
            )
        return validated


class StructuralLeakageValidator:
    """Enforce structural leakage rules without interpreting natural language."""

    @staticmethod
    def validate_evidence(
        evidence: EvidenceRecord | Mapping[str, Any],
    ) -> EvidenceRecord:
        return evidence if isinstance(evidence, EvidenceRecord) else EvidenceRecord.model_validate(evidence)

    @staticmethod
    def validate_model_context(
        context: CaseContext | Mapping[str, Any],
        *,
        data_file_paths: Iterable[str] = (),
    ) -> CaseContext:
        return ContextSchemaValidator.validate(context, data_file_paths=data_file_paths)


class DatasetContextProvenanceBuilder:
    """Build deterministic provenance for an approved dataset context pool."""

    @staticmethod
    def build(
        dataset_id: str,
        fact_index: Mapping[str, FactLocation],
        supporting_evidence: Mapping[str, Iterable[str]],
    ) -> DatasetContextProvenance:
        ordered_fact_ids = list(fact_index)
        if set(ordered_fact_ids) != set(supporting_evidence):
            raise ContextConstructionError(
                "fact index and supporting evidence must contain the same fact IDs"
            )
        return DatasetContextProvenance(
            dataset_id=dataset_id,
            fact_index=dict(fact_index),
            fact_provenance={
                fact_id: sorted(set(supporting_evidence[fact_id]))
                for fact_id in ordered_fact_ids
            },
        )


class CaseContextProvenanceBuilder:
    """Build the selected provenance subset with case-local fact locations."""

    @staticmethod
    def build(
        case_id: str,
        dataset_id: str,
        fact_index: Mapping[str, FactLocation],
        dataset_provenance: DatasetContextProvenance,
    ) -> CaseContextProvenance:
        unknown = sorted(set(fact_index) - set(dataset_provenance.fact_provenance))
        if unknown:
            raise ContextConstructionError(
                f"selected fact IDs have no dataset provenance: {unknown}"
            )
        return CaseContextProvenance(
            case_id=case_id,
            dataset_id=dataset_id,
            fact_index=dict(fact_index),
            fact_provenance={
                fact_id: dataset_provenance.fact_provenance[fact_id]
                for fact_id in fact_index
            },
        )


class DatasetContextBuilder:
    """Build the approved dataset context pool from explicit ContextFact records."""

    def build(
        self,
        dataset_id: str,
        sources: SourceCollection | Iterable[SourceRecord | Mapping[str, Any]],
        evidence_records: Iterable[EvidenceRecord | Mapping[str, Any]],
    ) -> DatasetContextBuildResult:
        _, evidence = EvidenceSourceCrossReferenceValidator.validate(
            dataset_id,
            sources,
            evidence_records,
        )

        definitions: dict[str, tuple[str, ReviewStatus, ContextFact]] = {}
        payload_identities: dict[str, str] = {}
        supporting_evidence: dict[str, set[str]] = {}

        for record in evidence:
            for fact in record.context_facts:
                payload = _canonical_payload(fact)
                existing = definitions.get(fact.fact_id)
                if existing is not None:
                    existing_payload, existing_review, _ = existing
                    if existing_payload != payload:
                        raise ContextConstructionError(
                            f"fact_id {fact.fact_id!r} has inconsistent field/value definitions"
                        )
                    if existing_review != fact.review_status:
                        raise ContextConstructionError(
                            f"fact_id {fact.fact_id!r} has inconsistent review_status values"
                        )
                else:
                    duplicate_id = payload_identities.get(payload)
                    if duplicate_id is not None and duplicate_id != fact.fact_id:
                        raise ContextConstructionError(
                            f"fact IDs {duplicate_id!r} and {fact.fact_id!r} have identical "
                            "canonical field/value definitions"
                        )
                    definitions[fact.fact_id] = (payload, fact.review_status, fact)
                    payload_identities[payload] = fact.fact_id

                if (
                    record.evidence_type == "context"
                    and record.eligible_for_context
                    and fact.review_status == "approved"
                ):
                    supporting_evidence.setdefault(fact.fact_id, set()).add(record.evidence_id)

        approved_ids = set(supporting_evidence)
        ordered_fact_ids = sorted(
            approved_ids,
            key=lambda fact_id: (
                _CONTEXT_FIELD_ORDER[definitions[fact_id][2].field],
                fact_id,
            ),
        )
        output: dict[str, Any] = {
            "physical_setting": None,
            "objects": [],
            "boundaries": [],
            "regions": [],
            "reference_directions": [],
            "operating_conditions": [],
            "geometry_assets": [],
        }
        fact_index: dict[str, FactLocation] = {}

        for fact_id in ordered_fact_ids:
            fact = definitions[fact_id][2]
            value = _validated_value(fact)
            if fact.field == "physical_setting":
                if output["physical_setting"] is not None:
                    raise ContextConstructionError(
                        "dataset context can contain only one approved physical_setting fact"
                    )
                output["physical_setting"] = value
                fact_index[fact_id] = FactLocation(field=fact.field, position=None)
            else:
                position = len(output[fact.field])
                output[fact.field].append(value)
                fact_index[fact_id] = FactLocation(field=fact.field, position=position)

        try:
            context = DatasetContext.model_validate(output)
        except ValueError as exc:
            raise ContextConstructionError(
                f"approved context facts conflict in the DatasetContext schema: {exc}"
            ) from exc

        provenance = DatasetContextProvenanceBuilder.build(
            dataset_id=dataset_id,
            fact_index=fact_index,
            supporting_evidence=supporting_evidence,
        )
        return DatasetContextBuildResult(context=context, provenance=provenance)


class CaseContextSelector:
    """Select explicitly annotated fact IDs without interpreting the question."""

    def select(
        self,
        dataset_context: DatasetContext | Mapping[str, Any],
        dataset_provenance: DatasetContextProvenance | Mapping[str, Any],
        selection: ContextSelection | Mapping[str, Any],
        *,
        data_file_paths: Iterable[str] = (),
    ) -> CaseContextSelectionResult:
        context = (
            dataset_context
            if isinstance(dataset_context, DatasetContext)
            else DatasetContext.model_validate(dataset_context)
        )
        provenance = (
            dataset_provenance
            if isinstance(dataset_provenance, DatasetContextProvenance)
            else DatasetContextProvenance.model_validate(dataset_provenance)
        )
        selection_record = (
            selection
            if isinstance(selection, ContextSelection)
            else ContextSelection.model_validate(selection)
        )
        if provenance.dataset_id != selection_record.dataset_id:
            raise ContextConstructionError(
                "context selection dataset_id does not match dataset context provenance"
            )
        actual_locations = {
            (location.field, location.position) for location in provenance.fact_index.values()
        }
        expected_locations = _context_locations(context)
        if actual_locations != expected_locations:
            raise ContextConstructionError(
                "dataset context provenance must cover every context fact exactly once"
            )

        unknown = sorted(set(selection_record.include_fact_ids) - set(provenance.fact_index))
        if unknown:
            raise ContextConstructionError(f"context selection references unknown fact IDs: {unknown}")

        selected_ids = set(selection_record.include_fact_ids)
        ordered_ids = sorted(
            selected_ids,
            key=lambda fact_id: (
                _CONTEXT_FIELD_ORDER[provenance.fact_index[fact_id].field],
                provenance.fact_index[fact_id].position
                if provenance.fact_index[fact_id].position is not None
                else -1,
                fact_id,
            ),
        )
        output: dict[str, Any] = {
            "physical_setting": None,
            "objects": [],
            "boundaries": [],
            "regions": [],
            "reference_directions": [],
            "operating_conditions": [],
            "geometry_assets": [],
        }
        selected_index: dict[str, FactLocation] = {}

        for fact_id in ordered_ids:
            location = provenance.fact_index[fact_id]
            if location.field == "physical_setting":
                if context.physical_setting is None:
                    raise ContextConstructionError(
                        f"fact index for {fact_id!r} points to an empty physical_setting"
                    )
                output["physical_setting"] = context.physical_setting
                selected_index[fact_id] = FactLocation(field=location.field, position=None)
                continue

            values = getattr(context, location.field)
            assert location.position is not None
            if location.position >= len(values):
                raise ContextConstructionError(
                    f"fact index for {fact_id!r} points outside {location.field}"
                )
            position = len(output[location.field])
            output[location.field].append(values[location.position].model_dump(mode="json"))
            selected_index[fact_id] = FactLocation(field=location.field, position=position)

        selected_context = CaseContext.model_validate(output)
        selected_context = ContextSchemaValidator.validate(
            selected_context, data_file_paths=data_file_paths
        )
        selected_provenance = CaseContextProvenanceBuilder.build(
            case_id=selection_record.case_id,
            dataset_id=selection_record.dataset_id,
            fact_index=selected_index,
            dataset_provenance=provenance,
        )
        return CaseContextSelectionResult(
            context=selected_context,
            provenance=selected_provenance,
        )
