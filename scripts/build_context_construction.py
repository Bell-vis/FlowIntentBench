"""Build dataset and case context resources from reviewed JSON evidence.

The script is deliberately data-driven: dataset facts, source identity, evidence,
case specifications, and case selections live in each dataset's ``construction``
directory.  Initial-case fixtures remain compatible without formal metadata;
formal cases opt into the existing question/metadata preflight.  This module
only loads the frozen schemas, invokes the existing builders/selectors, and
serializes their outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench import (
    BenchmarkCaseInput,
    CaseContextSelector,
    CaseConstructionMetadata,
    CaseConstructionMetadataValidator,
    ContextSelection,
    DatasetContextBuilder,
    DatasetManifest,
    EvidenceRecord,
    load_source_collection,
)
from flowintentbench.schema import DataMetadata
from flowintentbench.source_acquisition import acquire_source_assets


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _records(path: Path) -> list[EvidenceRecord]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"evidence resource must be a JSON array: {path}")
    return [EvidenceRecord.model_validate(item) for item in payload]


def _load_case_spec(
    case_dir: Path,
    *,
    require_formal_case: bool = False,
) -> tuple[str, ContextSelection, CaseConstructionMetadata | None]:
    """Load final question/selection and optionally validate formal metadata.

    ``initial_case`` remains a context-pipeline fixture and may omit metadata.
    A case with metadata is always preflighted; ``require_formal_case`` makes
    that metadata mandatory for a selected formal case.
    """

    question_payload = _read_json(case_dir / "scientific_question.json")
    if not isinstance(question_payload, dict):
        raise ValueError(
            f"scientific question resource must be a JSON object: "
            f"{case_dir / 'scientific_question.json'}"
        )
    question = question_payload.get("scientific_question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            f"scientific_question must be a non-empty string: "
            f"{case_dir / 'scientific_question.json'}"
        )

    selection = ContextSelection.model_validate(_read_json(case_dir / "context_selection.json"))
    metadata_path = case_dir / "case_construction_metadata.json"
    if not metadata_path.exists():
        if require_formal_case:
            raise FileNotFoundError(
                "formal case requires case_construction_metadata.json: "
                f"{metadata_path}"
            )
        return question, selection, None

    metadata = CaseConstructionMetadata.model_validate(_read_json(metadata_path))
    CaseConstructionMetadataValidator.validate(
        question,
        metadata,
        context_selection=selection,
    )
    return question, selection, metadata


def build_dataset(
    dataset_dir: Path,
    *,
    case_id: str = "initial_case",
    require_formal_case: bool = False,
) -> None:
    dataset_id = dataset_dir.name
    construction_dir = dataset_dir / "construction"
    case_dir = construction_dir / "cases" / case_id

    sources = load_source_collection(
        construction_dir / "sources.json",
        dataset_id=dataset_id,
        require_formal_source=True,
    )
    evidence: list[EvidenceRecord] = []
    for name in (
        "context_evidence.json",
        "operationalization_evidence.json",
        "finding_evidence.json",
    ):
        evidence.extend(_records(construction_dir / name))

    built = DatasetContextBuilder().build(dataset_id, sources, evidence)
    _write_json(construction_dir / "dataset_context.json", built.context.model_dump(mode="json"))
    _write_json(
        construction_dir / "dataset_context_provenance.json",
        built.provenance.model_dump(mode="json"),
    )

    question, selection, _ = _load_case_spec(
        case_dir,
        require_formal_case=require_formal_case,
    )
    selected = CaseContextSelector().select(
        built.context,
        built.provenance,
        selection,
        data_file_paths=[item.path for item in _case_flow_data(dataset_dir).data_files],
    )
    _write_json(case_dir / "case_context.json", selected.context.model_dump(mode="json"))
    _write_json(
        case_dir / "case_context_provenance.json",
        selected.provenance.model_dump(mode="json"),
    )

    flow_data = _case_flow_data(dataset_dir)
    case_input = BenchmarkCaseInput(
        scientific_question=question,
        flow_data=flow_data,
        case_context=selected.context,
    )
    _write_json(case_dir / "case_input.json", case_input.model_dump(mode="json"))


def _case_flow_data(dataset_dir: Path):
    metadata = DataMetadata.model_validate(_read_json(dataset_dir / "data_metadata.json"))
    manifest = DatasetManifest.model_validate(_read_json(dataset_dir / "dataset_manifest.json"))
    from flowintentbench.schema import DataFile, FlowData

    return FlowData(
        data_files=[DataFile(path=item.path, role=item.role) for item in manifest.files],
        data_metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("datasets"),
        help="dataset root containing dataset construction directories",
    )
    parser.add_argument(
        "--case-id",
        default="initial_case",
        help="case directory to build (default: initial_case)",
    )
    parser.add_argument(
        "--formal-case",
        action="store_true",
        help="require and preflight case_construction_metadata.json for the selected case",
    )
    parser.add_argument("dataset_ids", nargs="*", help="optional dataset IDs")
    parser.add_argument("--fetch-sources", action="store_true", help="acquire and verify construction/source_assets.json before building")
    parser.add_argument("--fetch-only", action="store_true", help="acquire source assets without constructing scientific artifacts")
    parser.add_argument("--prepare-data", action="store_true", help="run construction/source_import.json before building")
    parser.add_argument("--prepare-only", action="store_true", help="prepare source fields without building cases")
    args = parser.parse_args()
    directories = (
        [args.datasets_root / item for item in args.dataset_ids]
        if args.dataset_ids
        else sorted(item for item in args.datasets_root.iterdir() if item.is_dir())
    )
    for dataset_dir in directories:
        if (dataset_dir / "construction").is_dir():
            if args.fetch_sources or args.fetch_only:
                acquire_source_assets(dataset_dir)
            if args.prepare_data or args.prepare_only:
                from flowintentbench.source_preparation import prepare_source_data
                prepare_source_data(dataset_dir)
            if args.fetch_only or args.prepare_only:
                continue
            build_dataset(
                dataset_dir,
                case_id=args.case_id,
                require_formal_case=args.formal_case,
            )
            print(f"built {dataset_dir.name}")


if __name__ == "__main__":
    main()
