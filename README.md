# FlowIntentBench

**FlowIntentBench: Can LLM Agents Analyze Flow Fields to Answer Scientific Questions?**

FlowIntentBench evaluates how language-model agents turn scientific questions about
3D flow fields into analysis choices and evidence-supported findings. Each case
provides a question **Q**, numerical data **D**, and physical context **C**. An agent
reports its **operationalization O** (definitions, measurements, and decision rules)
and **findings F**. Reference branches pair accepted analyses with the findings
obtained by executing them, so different valid analyses need not produce identical
answers.

This anonymous source release includes the numerical datasets, case definitions,
frozen construction references, collection tools, and evaluation code. **Historical
model answers, trajectories, evaluation outputs, logs, credentials, and Git history
are excluded.** Running a command may create local files under `outputs/`; this
directory is ignored by Git.

## Benchmark at a glance

- **96 cases**, organized as **24 scientific question families × 4 conditions**.
- **15 numerical dataset inputs** from **14 parent source groups**. The two
  electrolyzer inputs share a source but have different fields and meshes.
- Seven task types: region analysis (8 families), field association (5), spatial
  heterogeneity (4), spatial extent (3), isosurface geometry (2), directional
  alignment (1), and advective flux (1).
- Each case concerns one spatial snapshot. Temporal analysis is outside this release.
- The original 28-case user-study definitions are retained separately; those IDs
  also occur in the 96-case manifest. Participant responses are not included.

| Condition | Analysis responsibility | Finding responsibility |
|---|---|---|
| O1–F1 | All principal decisions are specified | Report the specified finding roles |
| O2–F1 | One or two principal decisions are open | Same roles as O1–F1 |
| O3–F1 | Formulate the analysis under task constraints | Same roles as O1–F1 |
| O1–F2 | Same prescribed analysis as O1–F1 | Select a sufficient combination of roles, including mandatory roles |

O3 retains constraints imposed by the scientific target and scope. F2 is evaluated
against sufficient-role combinations rather than a hidden requirement to reproduce
every stored finding.

## Installation

Use **Python 3.12**. Linux is required for the primary network-isolated agent runtime.
The numerical verification and saved-answer tools can be used independently of that
runtime.

```bash
git clone https://github.com/Bell-vis/FlowIntentBench.git
cd FlowIntentBench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r runtime-requirements.txt
python -m pip install -e '.[construction,evaluation,test]'
python -m pip check
```

Alternatively, use `conda env create -f environment.yml`, `conda activate benchmark`,
and then install `-e '.[construction,evaluation,test]'`. The runtime pins NumPy,
SciPy, Matplotlib, Pydantic, PyYAML, and VTK in
[`runtime-requirements.txt`](runtime-requirements.txt). Raw data are included as
ordinary Git files; Git LFS and a separate dataset download are not required.
Allow approximately 0.8 GB for the working files, plus Git history and dependencies.

## Quick verification: no API key needed

Run from the repository root:

```bash
# Check all 96 cases, the original 28-case bindings, and every input data checksum.
# Also open all 15 dataset inputs with their declared VTK readers.
python scripts/verify_release.py --read-data

# Check frozen-reference preparation and saved-answer ingestion in temporary files.
# This makes zero API calls and leaves no benchmark results in the repository.
python scripts/smoke_release.py

# Regression tests for the shipped evaluation, numerical, and transport components.
python -m pytest -q -n 4
```

The ingestion smoke test intentionally leaves a synthetic answer unreviewed. Its
underlying evaluator returns **2 / INCOMPLETE**, which the smoke script checks as
expected behavior. It is not a scientific score or a completed model evaluation.

## Repository layout

```text
flowintentbench/                         Core schemas, readers, runtimes, and scorers
scripts/                                Collection, evaluation, and reference tools
scripts/reference_construction/         Numerical reference construction and audits
datasets/<dataset>/                     Numerical files, metadata, sources, recipes
datasets/expansion_v1/                  96-case construction manifest and case files
experiments/expansion_v1_development/    96-case evaluation manifest and frozen policies
experiments/userstudy/                  Original 28-case definitions
artifacts/reference/                    Frozen reference analyses required by cases
agents/                                 Agent configurations
runtime_profiles/                       Isolated, host-network, and local profiles
config/                                 Non-secret provider examples
scientific_review_profiles/             Scientific-review configuration
tests/                                  Self-contained release regression suite
docs/                                   Protocol and data provenance notes
```

The main entry is
[`experiments/expansion_v1_development/case_manifest.json`](experiments/expansion_v1_development/case_manifest.json).
Each row identifies the dataset, family, condition, question, context, ground truth,
and evaluation material, with SHA-256 bindings. References and ground truth are
**evaluator-only inputs**; do not expose them to the answering agent.

## Datasets

| Input directory | Families | Cases |
|---|---:|---:|
| AIDEAS_Blow_Mold | 3 | 12 |
| Blunt_Fin | 1 | 4 |
| Carotid | 1 | 4 |
| Combustor | 1 | 4 |
| Double_Fin | 1 | 4 |
| Electrolyzer | 1 | 4 |
| Electrolyzer_Electric | 1 | 4 |
| FireFlow | 1 | 4 |
| Kitchen | 1 | 4 |
| MHD_Turbulence | 2 | 8 |
| NASA_LOx_Post | 1 | 4 |
| Office | 1 | 4 |
| OpenFOAM_Tubes | 3 | 12 |
| Radiative_Mixing_Layer | 3 | 12 |
| Rayleigh_Taylor | 3 | 12 |
| **Total** | **24** | **96** |

Every dataset has `dataset_manifest.json` and `data_metadata.json`. Source records,
conversion recipes, selected source assets, and offline reconstruction byte ranges
are retained where supplied. See [data provenance](docs/data.md). Upstream source
attribution is preserved; anonymization does not remove third-party provenance or
change numerical arrays.

## Run an agent

The manuscript's primary condition uses `flow-python-v1`: isolated network access,
read-only case files at `/case`, a writable `/workspace`, and persistent Python state
within each independent trial. Install host `bubblewrap` (`bwrap`) and ensure user
namespaces are available before using this profile. Profiles ending in
`host-network` or `windows-local` describe different runtime conditions.

An explicit one-case collection entry is:

```bash
# Configure your provider endpoint in config/yiapi.toml and export YIAPI_API_KEY.
# The profile fixes the model identity; choose/edit a profile for your model.
python scripts/run_real_model_pilot.py \
  --case-manifest datasets/expansion_v1/case_manifest.json \
  --dataset Blunt_Fin --case blunt_fin_o1_f1 \
  --agent gpt-5.6-luna-xhigh-chat \
  --server-config config/yiapi.toml \
  --output-root outputs/my_collection
```

This entry produces a pilot collection, not a certification of the manuscript's
full experiment. It requires a compatible model endpoint and makes paid API calls.
The repository also provides multi-case orchestration scripts. Inspect each
script's `--help` and explicitly select its manifest, agent, and output location;
some retained maintenance tools expect historical inputs that are not distributed.

## Evaluate saved answers

The model-independent entry is `scripts/evaluate_model_answers.py`. It accepts
JSON, JSONL, or collection ledgers. A minimal JSONL row is:

```json
{"model_id":"your-model","case_id":"blunt_fin_o1_f1","trial":1,"answer":"## Operationalization\n...\n## Finding\n..."}
```

Use the actual unedited model response. Each `(model_id, case_id, trial)` must be
unique. `--answers` and `--collection` can be repeated; `--models`, `--cases`, and
`--trials` select the evaluation scope.

First check ingestion offline:

```bash
python scripts/evaluate_model_answers.py \
  --answers answers.jsonl --output outputs/offline_check \
  --offline --max-api-calls 0
```

A fresh offline run has no semantic reviewer decisions and normally returns exit
code **2**. To perform a bounded live review, copy `auth.example.json` to
`auth.local.json`, supply your own key locally, and configure a Responses-compatible
endpoint and model in `config/reviewer.example.toml`:

```bash
python scripts/evaluate_model_answers.py \
  --answers answers.jsonl --output outputs/review_run \
  --api-config config/reviewer.example.toml --auth-path auth.local.json \
  --reviewer-model YOUR_REVIEWER_MODEL --effort medium \
  --max-api-calls 8 --max-wall-seconds 300
```

The selected reviewer must support the request format and reasoning effort.
Authentication files are ignored by Git. The default network configuration uses
direct connections; edit `config/benchmark_network.toml` if an explicit proxy is
needed. This packaging verification does not submit live model or reviewer calls.

Reports are created locally under `OUTPUT/reports/`. Resume only with the same
input and evaluation contract. A new model, reference package, or protocol requires
a new output directory. `--require-identified` returns exit **3** when review is
complete but some applicable scores remain bounded rather than identified.

## Evaluation and manuscript correspondence

The manuscript distinguishes five metrics:

| Metric | Meaning | Code report key |
|---|---|---|
| S_O | Validity of the stated analysis | `o_score` |
| URS | Quality on dimensions designated for agent decisions | `urs` |
| R_F | Coverage of required findings or sufficient roles | `core_finding_recall` (F1), `finding_requirement_recall` (F2) |
| P_F | Support for distinct task-relevant reported findings | `finding_precision` |
| C_OF | Support under the agent's declared analysis | `c_score` |

Finding matching preserves analysis groups and reference-branch identity. Numerical
checks use frozen values, units, and policies. Missing required items, contradicted
claims, unverified claims, and inapplicable scores have different meanings.
The manuscript reports condition-balanced means and repeated-run variability;
coverage must accompany quality summaries.

**Version boundary:** the shipped generic evaluator identifies itself as
`rubric-evidence-v4.1`. It uses dimension decisions `MET`, `NOT_MET`, and
`UNVERIFIABLE`, and reports unresolved metric bounds. The supplied manuscript
also describes a later five-level (0–4) atomic-attribute assessment and its final
aggregation. This release does not claim that v4.1 reproduces that later assessment
or the published result tables. Historical reviewer decisions, supplementary output
packages, model responses, human-audit records, and participant responses are not
included. See [protocol notes](docs/protocol.md) before comparing scores.

The preserved manifest retains `DEVELOPMENT_EVALUATION_ONLY`, `formal_release=false`,
and its original calibration status. Packaging tests verify the shipped code and
inputs; they do not change scientific validation or release status.

## Anonymization and release scope

This snapshot starts a new Git history with anonymous author and committer metadata.
Original credentials, personal environment names, local path indexes, run outputs,
archives, manuscript author information, and historical experiment reports are
excluded. Only frozen scientific inputs needed by the shipped cases are retained.

The GitHub owner remains visible as `Bell-vis`; content anonymization cannot conceal
account ownership or activity outside this repository. Third-party datasets retain
their source terms and attribution. No blanket license is assigned to upstream data.
