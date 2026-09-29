# Protocol and implementation scope

The README follows the supplied manuscript's title, task formulation, condition
matrix, dataset inventory, and five-metric organization. Manuscript text and author
metadata are not copied into this repository.

## Frozen inputs

`experiments/expansion_v1_development/case_manifest.json` binds the 96 cases to typed
question inputs, construction metadata, ground truth, and evaluation material.
`datasets/expansion_v1/execution_evidence/` and the retained reference portfolio
supply precomputed evidence. These are evaluation inputs, not collected model
answers. Numerical source files and their checksums are unchanged.

The original 28-case definitions are retained in `experiments/userstudy/`. The 96-case
manifest should be used for the main benchmark. Do not concatenate the two manifests
and double-count shared IDs.

## Code map

- `flowintentbench/expansion_evaluation.py`: validate bound case artifacts.
- `flowintentbench/reference_packages.py`: assemble and validate frozen references.
- `flowintentbench/answer_collections.py`: ingest answers and collections.
- `flowintentbench/rubric_scoring.py`: current v4.1 answer interpretation and scoring.
- `flowintentbench/trusted_scoring.py`: evidence-aware metric utilities and bounds.
- `scripts/prepare_evaluation_references.py`: inspect all frozen reference packages.
- `scripts/evaluate_model_answers.py`: generic bounded/offline evaluation CLI.
- `scripts/reference_construction/`: separately executed numerical construction.
- `flowintentbench/python_runtime.py`: agent execution and isolation support.

## Assessment version

The manuscript describes normalized grades from a five-level atomic-attribute rubric
and N/A-aware, condition-balanced repeated-run aggregation. The supplied generic v4.1
code uses ternary dimension decisions and represents unresolved claims with metric
intervals. These are related evaluation surfaces, but they are not interchangeable.
Matching metric names alone does not establish numerical equivalence. Preserve the
protocol version, reviewer configuration, and reference identity with every new run.

This release provides no historical output packages or API receipts. Consequently,
exact replay of the manuscript's final tables and human-review corrections is not
available from this snapshot alone. Re-running a model or reviewer also depends on
its provider availability and can produce different responses.

## Release verification

`verify_release.py` checks case bindings, data SHA-256 values and sizes, and optionally
all VTK readers. `smoke_release.py` runs reference preparation from a different
working directory and checks the zero-call, incomplete saved-answer path. The test
suite covers scoring, evidence binding, uncertainty, geometry, numerical recipes,
reference extensions, concurrency, and transport recovery using local fixtures.
Historical integration tests tied to excluded output archives and retired construction
portfolios are outside the distributed test suite.

A successful local verification proves that the documented offline paths run with
the shipped inputs. It does not certify a live endpoint, host namespace permissions,
scientific coverage of every possible alternative method, or a completed benchmark.
