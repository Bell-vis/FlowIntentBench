# Tests

Run `python -m pytest -q` from the repository root. The suite validates all 96 cases
with in-memory reference-answer fixtures and checks atomic grades, URS membership,
source citations, reference matching, uncertainty, numeric recipes, aggregation,
review retry limits, and deterministic offline replay.

`python scripts/verify_release.py --read-data` checks all packaged numerical inputs
with their declared readers and file digests.
