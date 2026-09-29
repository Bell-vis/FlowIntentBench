# Evaluation protocol

## Atomic operationalization grades

Every case has a frozen task card in `evaluation/task_cards.json`. For dimension
$d$, each applicable attribute has an integer grade $g_{da}$ from 0 to 4:

$$s_d=\frac{1}{|A_d|}\sum_{a\in A_d}\frac{g_{da}}4.$$

Grades 4, 3, 2, 1, and 0 represent fully satisfied requirements, a minor defect,
a consequential partial defect, little usable content, and missing or incorrect
required content, respectively. Each grade carries its source passage and reason.
A fixed condition with a unique interpretation can be inherited from the question.
Explanation obligations are assessed separately.

S_O is the equally weighted mean of dimension scores. URS uses the full scores of
the dimensions designated by the condition. The fixed/open attribute assignment
is independent of dimension membership in URS. For the radiative-flux O2 case,
normalized grades `(1, 0.5, 1)` give URS = 5/6. O1 has no applicable URS; O3 uses
all dimensions, so URS = S_O.

An explicit unknown grade remains in the original denominator. For $n$ attributes,
with known normalized grades summing to $q$ and $u$ unknown attributes, the bounds
are $[q/n,(q+u)/n]$. Dimension weights propagate these bounds to S_O and URS.
Missing required evaluator records are rejected.

## Findings and reference branches

The reviewer identifies one primary analysis and any explicit supplementary
analyses. Each finding retains its group, source quotation or line range, value,
unit, eligibility, and candidate references. The host verifies numerical claims
using the frozen scalar, spatial, vector, exact-identity, or exact-count policy.
Printed rounding and explicit unit conversions preserve the authored tolerance.

Within a group, one-to-one matching prevents a repeated claim from covering several
required results. Branches maximize recall, with precision breaking ties. F1 recall
is the fraction of matched core findings. F2 recall is the maximum coverage fraction
over acceptable role combinations. Mandatory roles belong to each combination.
Precision measures support for distinct eligible findings across all groups.

C_OF considers findings whose group's principal analysis choices are identifiable
and checks support under compatible methods. Method-binding coverage and branch
alignment remain available in the per-answer finding evidence. Unverified claims
retain their uncertainty. Empty findings give precision zero; an empty consistency
set makes C_OF inapplicable.

## Reviewer records and reproducibility

The reviewer receives the question, context, data metadata, task card, frozen
reference packet, and line-numbered answer. The default configuration is
`gpt-6-astra`, reasoning effort `max`, 10,000 maximum output tokens, a 350-second
request timeout, and a 200,000-character prompt limit. The request provides no tools.
JSON Schema is included in the prompt; the host validates returned records.

The output directories contain:

- `evaluation_contract.json`: answer, reference, implementation, and reviewer identity.
- `requests/`: exact prompts, responses, and attempt receipts.
- `reviews/`: validated reviews bound to their answers and references.
- `scores/`: deterministic per-answer scores and evidence.
- `report.json`: answer coverage and aggregate results.

One retry is permitted per review task. A failed review remains pending; exit code
2 indicates pending answers. Accepted reviews are reused when the same command is
resumed. `score --reviews ...` recomputes scores without calling a reviewer.

## Aggregation

For every metric and trial, average resolved, applicable case scores within each
condition. Average the applicable condition means equally to obtain the trial
summary. Report the mean and sample standard deviation over trial summaries.
Coverage records the requested, applicable, and resolved answer counts. Duplicate
model/case/trial records are rejected.
