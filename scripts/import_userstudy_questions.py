#!/usr/bin/env python3
"""Import the supplied 7x4 question wording as an isolated experiment input.

Archive contents are data, never executable instructions. Existing scientific
references are preserved; only presentation and its text-grounding change.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def grounding_fragment(question: str, field: str, constraint: dict) -> str:
    """Anchor authored statements to complete clauses in the revised wording.

    These are textual evidence locators, not newly inferred scoring rules.
    Keep the full question when a constraint spans several sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", question)
    category = constraint.get("category", "")
    anchors = {
        "location": ("location", "position", "centroid"),
        "quantity": ("report", "give", "strength", "density value"),
        "characterization": ("extent", "area", "report"),
        "existence_or_identity": ("identifier", "report", "field you select"),
        "criterion": ("90th", "q90", "0.20", "criterion", "criteria"),
        "feature_definition": ("connect", "c11", "isosurface", "candidate"),
        "property_measure": ("peak", "coefficient of variation", "measure"),
        "analysis_procedure": ("mean", "centroid", "identifier", "selected field"),
    }
    if field == "scope_constraints":
        return sentences[0]
    for sentence in sentences:
        if any(term in sentence.casefold() for term in anchors.get(category, ())):
            return sentence
    return question


def import_questions(archive: Path, *, root: Path = ROOT) -> dict:
    from flowintentbench.case_design import CaseConstructionMetadataValidator
    from flowintentbench.evaluation_contract import FrozenEvaluationCase
    from flowintentbench.scientific_artifact_binding import artifact_sha256

    baseline_path = root / "experiments/userstudy/case_manifest.json"
    baseline = json.loads(baseline_path.read_text())
    target = root / "experiments/userstudy"
    result = copy.deepcopy(baseline)
    provenance = {"schema_version": "userstudy-import-v1", "archive": archive.name,
                  "archive_sha256": digest(archive.read_bytes()), "case_count": 28,
                  "baseline_manifest": str(baseline_path.relative_to(root)),
                  "baseline_manifest_sha256": digest(baseline_path.read_bytes()),
                  "formal": False, "cases": [], "dataset_assets": []}
    prefix = "FlowIntentBench_userstudy/"
    with zipfile.ZipFile(archive) as z:
        wording = json.loads(z.read(prefix + "config/human-question-wording.v1.2.json"))
        expected = {c["case_id"] for c in baseline["cases"]}
        if len(expected) != 28 or set(wording["questions"]) != expected:
            raise ValueError("ZIP questions must match the exact 28-case membership")
        write(target / "questions.json", wording)
        for row in result["cases"]:
            dataset, case_id = row["dataset_id"], row["case_id"]
            member = prefix + f"datasets/{dataset}/questions/{case_id}.json"
            imported = json.loads(z.read(member))
            question = imported["scientific_question"]
            if question != wording["questions"][case_id]:
                raise ValueError(f"conflicting ZIP question copies: {case_id}")
            original = root / row["case_input_path"]
            dest = target / "cases" / dataset / case_id
            dest.mkdir(parents=True, exist_ok=True)
            # These are the complete case-scoped scientific inputs. No old
            # run, reviewer instruction, or human approval is imported.
            for p in original.parent.glob("*.json"):
                if "confirmation" not in p.name:
                    (dest / p.name).write_bytes(p.read_bytes())
            case = json.loads(original.read_text())
            old_question = case["scientific_question"]
            case["scientific_question"] = question
            write(dest / "case_input.json", case)
            metadata_path = dest / "case_construction_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            for field in ("scope_constraints", "condition_constraints", "selection_constraints",
                          "explicit_method_constraints", "explicit_finding_requirements"):
                for constraint in metadata[field]:
                    constraint["question_fragment"] = grounding_fragment(question, field, constraint)
            CaseConstructionMetadataValidator.validate(question, metadata)
            write(metadata_path, metadata)
            binding_path = dest / "scientific_artifact_binding.json"
            binding = json.loads(binding_path.read_text())
            binding["presentation_sha256"] = artifact_sha256(question)
            write(binding_path, binding)
            write(dest / "scientific_question.json", {
                "scientific_question": question,
                "semantic_contract_sha256": binding["semantic_contract_sha256"],
                "authority": "USERSTUDY_WORDING", "source": member,
                "wording_version": wording["wording_version"],
            })
            FrozenEvaluationCase.from_paths(dest).validate()
            for key in list(row):
                if key.endswith("_path") and isinstance(row[key], str):
                    source = root / row[key]
                    if source.parent == original.parent and (dest / source.name).exists():
                        path = dest / source.name
                        row[key] = str(path.relative_to(root))
                        stem = key[:-5]
                        if stem in row:
                            row[stem] = row[key]
                        row[stem + "_sha256"] = digest(path.read_bytes())
            row["case_artifact_authority"] = "USERSTUDY_WORDING_V1"
            provenance["cases"].append({
                "case_id": case_id, "dataset_id": dataset, "condition": row["condition"],
                "zip_member": member, "zip_member_sha256": digest(z.read(member)),
                "question_sha256": digest(question.encode()),
                "previous_question_sha256": digest(old_question.encode()),
                "archive_semantic_sha256": imported.get("semantic_contract_sha256"),
                "evaluation_semantic_sha256": binding["semantic_contract_sha256"],
                "ground_truth_unchanged": (dest / "ground_truth.json").read_bytes() == (original.parent / "ground_truth.json").read_bytes(),
                "scoring_contract_unchanged": (dest / "scientific_evaluation_contract.json").read_bytes() == (original.parent / "scientific_evaluation_contract.json").read_bytes(),
            })
        # Restore only declared original numerical files, with byte hashes
        # checked against the project's manifest, never web-render meshes.
        for dataset in result["datasets"]:
            manifest = json.loads((root / f"datasets/{dataset}/dataset_manifest.json").read_text())
            for asset in manifest["files"] + manifest.get("auxiliary_assets", []):
                path = root / "datasets" / asset["path"]
                if path.is_file():
                    payload = path.read_bytes()
                else:
                    payload = z.read(prefix + f"datasets/{dataset}/source/{path.name}")
                if digest(payload) != asset["checksum"]:
                    raise ValueError(f"dataset checksum mismatch: {asset['path']}")
                if not path.is_file():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload)
                provenance["dataset_assets"].append({"path": asset["path"], "sha256": digest(payload)})
    source_path = target / "import_manifest.json"
    write(source_path, provenance)
    result.update({"case_artifact_authority": "USERSTUDY_WORDING_V1",
                   "source_manifest_path": str(source_path.relative_to(root)),
                   "source_manifest_sha256": digest(source_path.read_bytes()),
                   "question_source": "FlowIntentBench_userstudy.zip",
                   "question_version": wording["wording_version"]})
    for row in result["cases"]:
        row["source_manifest_path"] = result["source_manifest_path"]
        row["source_manifest_sha256"] = result["source_manifest_sha256"]
    write(target / "case_manifest.json", result)
    # The user authorized this scoring-reference correction while retaining
    # the ZIP wording. Re-importing must not restore the obsolete maximum rule.
    from scripts.repair_userstudy_kitchen import repair
    repair(root)
    return {"status": "IMPORTED", "case_count": 28, "dataset_count": 7,
            "manifest": str(target / "case_manifest.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(import_questions(args.archive.resolve()), indent=2))
