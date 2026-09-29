# Release regression tests

Run `python -m pytest -q -n 4` from the repository root. These tests exercise the shipped numerical and evaluation components with local inputs. They do not contact paid reviewer services. Historical integration tests that require excluded experiment outputs or retired construction portfolios are not part of this release.

Run `python scripts/verify_release.py --read-data` and `python scripts/smoke_release.py` for the complete shipped input inventory and CLI smoke check.
