# FlowIntentBench

Official Repository for ICLR 2027 "**43241**" (Under Review)

**FlowIntentBench: Can LLM Agents Analyze Flow Fields to Answer Scientific Questions?**

FlowIntentBench measures how LLM agents choose scientific analyses and report
supported findings from 3D flow fields. This repository provides the datasets,
case definitions, frozen references, and a reproducible implementation of the
paper's evaluation protocol.

## Method

Each case supplies a scientific question **Q**, flow-field data **D**, and physical
context **C**. The agent returns an **operationalization O** describing its analysis
choices and **findings F** describing the resulting evidence.

Evaluation has three steps:

1. **Assess the analysis.** Grade each task-card attribute from 0 to 4, normalize by
   four, and average within dimensions and then across dimensions.
2. **Verify the findings.** Extract claims with source citations and match them to
   frozen reference branches. The host checks numerical values, units, and the
   case's reporting tolerances. Branch selection maximizes finding recall, with
   precision as the tie-break.
3. **Check method–result consistency.** Verify whether the findings are supported
   under the agent's declared analysis, then aggregate scores by condition and run.

| Metric | Meaning |
|---|---|
| S_O | Overall operationalization quality |
| URS | Quality of the condition-designated analysis dimensions |
| R_F | Coverage of required findings or sufficient finding roles |
| P_F | Support for reported scientific findings |
| C_OF | Consistency between declared analysis and reported findings |

## Benchmark

The benchmark contains **96 cases in 24 scientific question families**, using
**15 dataset inputs from 14 parent source groups**. Its task cards specify
**972 atomic attributes: 694 fixed and 278 open**.

| Condition | Analysis choices | Findings |
|---|---|---|
| O1–F1 | Principal decisions specified | Required roles specified |
| O2–F1 | One or two principal decisions open | Same roles as O1–F1 |
| O3–F1 | Agent formulates the analysis under task constraints | Same roles as O1–F1 |
| O1–F2 | Same analysis as O1–F1 | Agent selects a sufficient combination of roles |

Task families cover region analysis, field association, spatial heterogeneity,
spatial extent, isosurface geometry, directional alignment, and advective flux.
The original 28-case user-study definitions are included in `experiments/userstudy/`.

## Install

Use Python 3.12 and run these commands from the repository root:

```bash
git clone https://github.com/Bell-vis/FlowIntentBench.git
cd FlowIntentBench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r runtime-requirements.txt
python -m pip install -e '.[construction,evaluation,test]'
```

The numerical data are included in the repository. The working files occupy
approximately 0.8 GB. Source records, reader settings, and conversion information
are described in [Data](docs/data.md).

## Reproduce the evaluation

### 1. Validate the data and prepare the evaluator

```bash
python scripts/verify_release.py --read-data
python scripts/evaluate.py prepare --output outputs/prepared
```

These commands validate the 15 numerical inputs, load all 96 cases and their task
cards, and export the structured reviewer schema.

### 2. Supply model answers

Save the model's responses as `answers.jsonl`, with one record per case and run:

```json
{"model_id":"my-model","case_id":"blunt_fin_o1_f1","trial":1,"answer":"## Operationalization\n...\n## Finding\n..."}
```

The answering agent receives the question, numerical data, and physical context.
Reference analyses and task-card grades are used by the evaluator. Use trial IDs
`1`, `2`, and `3` for the paper's repeated-run setup.

### 3. Run the reviewer and compute scores

Configure a Responses-compatible endpoint and API key in your shell:

```bash
export FLOWINTENT_API_BASE="https://YOUR_PROVIDER/v1"
export OPENAI_API_KEY="YOUR_API_KEY"

python scripts/evaluate.py evaluate \
  --answers answers.jsonl \
  --output outputs/evaluation \
  --reviewer-model gpt-6-astra --effort max \
  --max-output-tokens 10000 --timeout 350 \
  --max-prompt-chars 200000 --max-api-calls 8
```

The reviewer extracts methods and findings, assigns atomic grades, and supplies
semantic reference matches. The host validates the records, performs numerical
checks, and computes the five metrics. Set `--max-api-calls` to the budget for your
answer set; repeat the same command to resume. Each review task permits one retry.

### 4. Recompute scores offline

Saved reviews can be replayed with no API calls:

```bash
python scripts/evaluate.py score \
  --answers answers.jsonl \
  --reviews outputs/evaluation/reviews \
  --output outputs/reproduced
```

`report.json` contains per-answer scores, coverage, condition means, run means,
and the mean and sample standard deviation across runs. Atomic grades, source
citations, reference matches, and numerical checks accompany each score.
Generated results are written under `outputs/`, which is excluded from Git.

## Scoring rules

S_O averages all applicable dimension scores. URS averages the complete scores of
the designated dimensions, including fixed attributes within mixed dimensions.
For O3, URS and S_O use the same dimension set.

F1 recall measures matched core findings. F2 recall measures coverage of authored
sufficient-role combinations. Matching is one-to-one within a reference branch.
Supplementary analyses contribute to precision and consistency; recall concerns
the primary answer.

A demonstrated omission receives zero. An unassessable attribute retains its
predefined denominator and produces a score interval. Point-score summaries use
resolved, applicable values. Cases have equal weight within each condition;
condition means have equal weight within a run. See [Protocol](docs/protocol.md)
for the formulas and record format.

## Repository layout

```text
datasets/                              Numerical inputs, metadata, and case construction
experiments/expansion_v1_development/   96-case manifest and frozen evaluation material
experiments/userstudy/                  Original 28-case definitions
evaluation/task_cards.json             Atomic attributes and explanation obligations
artifacts/reference/                   Executed reference analyses
flowintentbench/paper_evaluation.py     Atomic scoring and paper-level aggregation
flowintentbench/finding_scoring.py      Finding matching and O–F consistency
flowintentbench/numeric_verification.py Frozen numerical verification rules
scripts/evaluate.py                    Preparation, review, and offline scoring CLI
scripts/verify_release.py              Case and numerical-input validation
tests/                                 Scoring and numerical regression tests
```

## Tests

```bash
python -m pytest -q
```

The tests cover reference-answer scoring for all 96 cases, five-level grades,
mixed-dimension URS, null propagation, finding verification, source binding,
condition-balanced aggregation, retry handling, and offline score reproduction.
