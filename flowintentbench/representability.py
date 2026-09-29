"""Small machine-readable registry for SCQ evaluation representability."""

from __future__ import annotations

import inspect
from functools import partial
from typing import Any, Callable, Mapping


REPRESENTABILITY_KEYS = (
    "o_schema_available",
    "materializer_available",
    "finding_extractor_available",
    "deterministic_handlers",
    "semantic_handler_available",
    "valid_unenumerated_escalation",
    "uncertain_escalation",
)

DEFAULT_EVALUATION_REPRESENTABILITY_CONTRACT: dict[str, Any] = {
    "o_schema_available": True,
    "materializer_available": True,
    "finding_extractor_available": True,
    "deterministic_handlers": ["exact", "scalar_numeric", "spatial_tolerance"],
    "semantic_handler_available": True,
    "valid_unenumerated_escalation": True,
    "uncertain_escalation": True,
}


# This is deliberately a small addressability registry, rather than a new
# execution framework.  Values are callables so a production contract cannot
# pass merely because it contains a descriptive string.  Most routes are
# intentionally thin adapters: Q3 checks that the route is registered; the
# scientific quality of a future invocation remains a calibration concern.
def _materialize_case_route(proposal: Any, deterministic_executor: Any) -> Any:
    from .srac import materialize_branch

    return materialize_branch(proposal, deterministic_executor)


def _extract_atomic_findings_route(*args: Any, **kwargs: Any) -> Any:
    from .evaluation_integration import extract_atomic_findings

    return extract_atomic_findings(*args, **kwargs)


def _deterministic_verifier_route(
    predicted: Any, reference: Any, *, allowed_modes: tuple[str, ...],
    unit_converter: Any = None, explicit_policy: Any = None,
) -> bool:
    from .evaluator import EvaluationConfigurationError, finding_verification_mode, verify_finding_value

    if reference.value is None or finding_verification_mode(reference, explicit_policy).value not in allowed_modes:
        raise EvaluationConfigurationError("deterministic handler requires an explicit compatible GT value rule")
    return verify_finding_value(predicted, reference, unit_converter=unit_converter, explicit_policy=explicit_policy)


# Handler names declare a numerical/identity contract; they must never silently
# fall back to semantic-only acceptance when the GT tolerance is absent.
_exact_verifier_route = partial(_deterministic_verifier_route, allowed_modes=("exact_discrete_numeric", "exact_identity"))
_scalar_numeric_verifier_route = partial(_deterministic_verifier_route, allowed_modes=("scalar_tolerance",))
_spatial_tolerance_verifier_route = partial(_deterministic_verifier_route, allowed_modes=("spatial_euclidean",))


def _scientific_adjudication_route(proposal: Any, adjudicator: Mapping[str, Any], scientific_view: Any, invoke: Any) -> Any:
    from .proxy_expert import prepare_adjudicator_invocation

    return prepare_adjudicator_invocation(proposal, adjudicator, scientific_view, invoke)


def _valid_unenumerated_route(value: Any = None) -> dict[str, Any]:
    return {"status": "ESCALATE_TO_CURATOR", "reason": "VALID_UNENUMERATED", "payload": value}


def _uncertain_route(value: Any = None) -> dict[str, Any]:
    return {"status": "PENDING_ADJUDICATION", "reason": "UNCERTAIN", "payload": value}


def _smoke_materializer(route: Callable[..., Any]) -> bool:
    from .srac import SRACProposal, MaterializedBranch

    proposal = SRACProposal("registry-smoke", "O1-F1", "producer", "family")
    result = route(proposal, lambda _proposal: {"execution": {}})
    return isinstance(result, MaterializedBranch)


def _smoke_extractor(route: Callable[..., Any]) -> bool:
    result = route("## Finding\nThe region is located at (0, 0, 0).")
    return getattr(result, "status", None) == "EXTRACTED"


def _smoke_verifier(route: Callable[..., Any], *, kind: str) -> bool:
    from .evaluator import PredictedAtomicFinding
    from .ground_truth import ReferenceFinding, FindingRequirementCategory, FindingImportance, VerificationSpec

    # Include a nearby accepted value and a rejected value, so an always-True
    # adapter or semantic-only verifier cannot satisfy numerical representability.
    value, accepted, rejected, verification, policy = {
        "exact": (1, 1, 2, None, "exact_discrete_numeric"),
        "scalar_numeric": (1.0, 1.05, 1.5, VerificationSpec(absolute_tolerance=0.1), None),
        "spatial_tolerance": ([0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [1.0, 0.0, 0.0], VerificationSpec(spatial_tolerance=0.1), None),
    }[kind]
    reference = ReferenceFinding(finding_id="registry-smoke", category=FindingRequirementCategory.QUANTITY, statement="Registry value probe.", importance=FindingImportance.CORE, value=value, verification=verification)
    good = PredictedAtomicFinding("registry-good", "Compatible value.", value=accepted)
    bad = PredictedAtomicFinding("registry-bad", "Incompatible value.", value=rejected)
    kwargs = {"explicit_policy": policy} if policy else {}
    return route(good, reference, **kwargs) is True and route(bad, reference, **kwargs) is False


def _smoke_semantic(route: Callable[..., Any]) -> bool:
    # The real adapter is smoke-tested without a model call; the invocation
    # callback is the existing production orchestration seam.
    result = route({"producer_agent_id": "producer"}, {"agent_id": "adjudicator"}, {"question": "visible", "candidate_F": [], "materialized_G_of_O": {"findings": []}}, lambda _view: {"status": "SMOKE"})
    return result == {"status": "SMOKE"}


def _smoke_escalation(route: Callable[..., Any]) -> bool:
    return isinstance(route(None), Mapping)


def _descriptor(handler_id: str, kind: str, route: Callable[..., Any], binding_target: str, smoke: Callable[[Callable[..., Any]], bool]) -> dict[str, Any]:
    return {"handler_id": handler_id, "handler_kind": kind, "callable": route, "production_binding": True, "binding_target": binding_target, "smoke_testable": True, "smoke": smoke}


REPRESENTABILITY_HANDLER_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
    "materializers": {
        "deterministic_case_materializer_v1": _descriptor("deterministic_case_materializer_v1", "materializer", _materialize_case_route, "flowintentbench.srac.materialize_branch", _smoke_materializer),
    },
    "finding_extractors": {
        "atomic_finding_extractor_v1": _descriptor("atomic_finding_extractor_v1", "finding_extractor", _extract_atomic_findings_route, "flowintentbench.evaluation_integration.extract_atomic_findings", _smoke_extractor),
    },
    "deterministic_verifiers": {
        "exact": _descriptor("exact", "deterministic_verifier", _exact_verifier_route, "flowintentbench.evaluator.verify_finding_value", partial(_smoke_verifier, kind="exact")),
        "scalar_numeric": _descriptor("scalar_numeric", "deterministic_verifier", _scalar_numeric_verifier_route, "flowintentbench.evaluator.verify_finding_value", partial(_smoke_verifier, kind="scalar_numeric")),
        "spatial_tolerance": _descriptor("spatial_tolerance", "deterministic_verifier", _spatial_tolerance_verifier_route, "flowintentbench.evaluator.verify_finding_value", partial(_smoke_verifier, kind="spatial_tolerance")),
    },
    "semantic_adjudicators": {
        "scientific_adjudication_v1": _descriptor("scientific_adjudication_v1", "semantic_adjudicator", _scientific_adjudication_route, "flowintentbench.proxy_expert.prepare_adjudicator_invocation", _smoke_semantic),
    },
    "escalation_routes": {
        "valid_unenumerated": _descriptor("valid_unenumerated", "escalation_route", _valid_unenumerated_route, "flowintentbench.representability._valid_unenumerated_route", _smoke_escalation),
        "uncertain": _descriptor("uncertain", "escalation_route", _uncertain_route, "flowintentbench.representability._uncertain_route", _smoke_escalation),
    },
}


def get_handler_descriptor(category: str, handler_id: Any) -> Mapping[str, Any] | None:
    value = REPRESENTABILITY_HANDLER_REGISTRY.get(category, {}).get(handler_id) if isinstance(handler_id, str) else None
    if isinstance(value, Mapping):
        return value
    # A raw callable can be injected by a test or an extension, but it is not
    # considered production-bound until it carries the explicit descriptor.
    if callable(value):
        return {"handler_id": handler_id, "callable": value, "production_binding": False}
    return None


def resolve_handler(category: str, handler_id: Any) -> Callable[..., Any] | None:
    """Resolve one canonical route ID from the explicit registry."""

    if not isinstance(handler_id, str) or not handler_id.strip():
        return None
    descriptor = get_handler_descriptor(category, handler_id)
    route = descriptor.get("callable") if descriptor else None
    return route if callable(route) else None


def representability_handler_registry_snapshot() -> dict[str, list[str]]:
    return {category: sorted(routes) for category, routes in REPRESENTABILITY_HANDLER_REGISTRY.items()}


def _normalize_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the frozen nested production contract to registry fields.

    The nested form is retained in the returned ``contract`` for audit
    provenance, while the flat aliases keep the existing SCQ checks small and
    backwards compatible with older fixtures.
    """

    contract = dict(value)
    materialization = contract.get("materialization")
    if isinstance(materialization, Mapping):
        contract.setdefault("materializer_available", materialization.get("available"))
    finding_representation = contract.get("finding_representation")
    if isinstance(finding_representation, Mapping):
        contract.setdefault("finding_extractor_available", finding_representation.get("available"))
    deterministic = contract.get("deterministic_verification")
    if isinstance(deterministic, Mapping):
        handlers = deterministic.get("registered_handlers", ())
        contract.setdefault("deterministic_handlers", list(handlers) if isinstance(handlers, (list, tuple, set)) else handlers)
    semantic = contract.get("semantic_adjudication")
    if isinstance(semantic, Mapping):
        contract.setdefault("semantic_handler_available", semantic.get("handler_available"))
    return contract


def validate_representability_contract(
    value: Mapping[str, Any] | None,
    *,
    allow_default: bool = True,
) -> dict[str, Any]:
    """Return deterministic status/failure codes without calibration inputs.

    ``allow_default`` exists only for legacy synthetic/unit fixtures.  The
    production SCQ path passes ``False`` so a missing contract is an explicit
    failure rather than an all-true fallback.
    """

    if value is None:
        if not allow_default:
            return {
                "status": "FAIL",
                "failure_codes": ["EVALUATION_REPRESENTABILITY_CONTRACT_MISSING"],
                "contract": None,
                "contract_present": False,
            }
        contract = dict(DEFAULT_EVALUATION_REPRESENTABILITY_CONTRACT)
        contract_present = False
    elif isinstance(value, Mapping):
        contract = _normalize_contract(value)
        contract_present = True
    else:
        return {
            "status": "FAIL",
            "failure_codes": ["EVALUATION_REPRESENTABILITY_CONTRACT_INVALID"],
            "contract": value,
            "contract_present": False,
        }
    failures: list[str] = []
    resolved_routes: list[dict[str, Any]] = []

    def validate_route(category: str, handler_id: Any, *, smoke: bool = True) -> None:
        descriptor = get_handler_descriptor(category, handler_id)
        if descriptor is None:
            failures.append("REPRESENTABILITY_HANDLER_UNRESOLVED")
            resolved_routes.append({"category": category, "handler_id": handler_id, "status": "UNRESOLVED"})
            return
        route = descriptor.get("callable")
        if not callable(route):
            failures.append("REPRESENTABILITY_HANDLER_UNRESOLVED")
            resolved_routes.append({"category": category, "handler_id": handler_id, "status": "UNRESOLVED"})
            return
        if descriptor.get("production_binding") is not True or not isinstance(descriptor.get("binding_target"), str) or not descriptor.get("binding_target") or not descriptor.get("binding_target", "").casefold().startswith("flowintentbench."):
            failures.append("REPRESENTABILITY_HANDLER_NOT_PRODUCTION_BOUND")
            resolved_routes.append({"category": category, "handler_id": handler_id, "status": "NOT_PRODUCTION_BOUND"})
            return
        try:
            signature = str(inspect.signature(route))
        except (TypeError, ValueError):
            failures.append("REPRESENTABILITY_HANDLER_SIGNATURE_INVALID")
            resolved_routes.append({"category": category, "handler_id": handler_id, "status": "SIGNATURE_INVALID"})
            return
        smoke_result = "NOT_RUN"
        if smoke and descriptor.get("smoke_testable") is True:
            smoke_fn = descriptor.get("smoke")
            try:
                if not callable(smoke_fn) or smoke_fn(route) is not True:
                    raise RuntimeError("smoke returned false")
                smoke_result = "PASS"
            except Exception as exc:  # pragma: no cover - diagnostic boundary
                failures.append("REPRESENTABILITY_HANDLER_SMOKE_FAILED")
                smoke_result = f"FAIL:{type(exc).__name__}"
        resolved_routes.append({"category": category, "handler_id": handler_id, "status": "PASS", "smoke": smoke_result, "binding_target": descriptor.get("binding_target"), "callable_signature": signature})
    nested_contract = any(key in contract for key in ("materialization", "finding_representation", "deterministic_verification", "semantic_adjudication"))
    # Production contracts use the nested route-bearing schema.  Legacy flat
    # fixtures remain accepted only for explicitly scoped synthetic tests.
    if nested_contract:
        required_sections = ("materialization", "finding_representation", "deterministic_verification", "semantic_adjudication")
        if any(not isinstance(contract.get(section), Mapping) for section in required_sections):
            failures.append("EVALUATION_REPRESENTABILITY_CONTRACT_INVALID")
        if contract.get("o_schema_available") is not True:
            failures.append("EVALUATION_REPRESENTABILITY_CONTRACT_INVALID")
        for section in ("materialization", "finding_representation", "deterministic_verification", "semantic_adjudication"):
            item = contract.get(section)
            if isinstance(item, Mapping) and item.get("available") is False:
                failures.append({
                    "materialization": "G_OF_O_MATERIALIZER_MISSING",
                    "finding_representation": "ATOMIC_FINDING_PATH_MISSING",
                    "deterministic_verification": "DETERMINISTIC_VERIFICATION_ROUTE_MISSING",
                    "semantic_adjudication": "SEMANTIC_ADJUDICATION_ROUTE_MISSING",
                }[section])
        materialization = contract.get("materialization")
        finding = contract.get("finding_representation")
        deterministic = contract.get("deterministic_verification")
        semantic = contract.get("semantic_adjudication")
        if isinstance(materialization, Mapping) and (materialization.get("available") is not True or not materialization.get("handler_id")):
            failures.append("EVALUATION_REPRESENTABILITY_CONTRACT_INVALID")
        if isinstance(finding, Mapping) and (finding.get("available") is not True or not finding.get("extractor_id")):
            failures.append("EVALUATION_REPRESENTABILITY_CONTRACT_INVALID")
        if isinstance(semantic, Mapping) and (semantic.get("required", True) is not True and semantic.get("required") is not False or not semantic.get("handler_id")):
            failures.append("EVALUATION_REPRESENTABILITY_CONTRACT_INVALID")
        if isinstance(materialization, Mapping):
            before = len(failures); validate_route("materializers", materialization.get("handler_id"));
            if len(failures) > before and failures[-1] == "REPRESENTABILITY_HANDLER_NOT_PRODUCTION_BOUND":
                pass
        if isinstance(finding, Mapping):
            validate_route("finding_extractors", finding.get("extractor_id"))
        if isinstance(semantic, Mapping) and bool(semantic.get("required", True)):
            prior = len(failures); validate_route("semantic_adjudicators", semantic.get("handler_id"))
            # A missing semantic route has its semantic-specific code in Q3.
            if len(failures) > prior and failures[-1] in {"REPRESENTABILITY_HANDLER_UNRESOLVED", "REPRESENTABILITY_HANDLER_NOT_PRODUCTION_BOUND", "REPRESENTABILITY_HANDLER_SIGNATURE_INVALID", "REPRESENTABILITY_HANDLER_SMOKE_FAILED"}:
                failures.append("SEMANTIC_ADJUDICATION_ROUTE_MISSING")
        if isinstance(semantic, Mapping) and bool(semantic.get("required", True)) and semantic.get("handler_available") is not True:
            failures.append("SEMANTIC_ADJUDICATION_ROUTE_MISSING")
        if isinstance(deterministic, Mapping):
            required_types = deterministic.get("required_claim_types")
            declared = deterministic.get("registered_handlers")
            if not isinstance(required_types, (list, tuple, set)) or not isinstance(declared, (list, tuple, set)):
                failures.append("EVALUATION_REPRESENTABILITY_CONTRACT_INVALID")
            else:
                declared_ids = {str(item) for item in declared}
                for handler_id in declared_ids:
                    validate_route("deterministic_verifiers", handler_id)
                handler_for_type = {"exact": "exact", "scalar_numeric": "scalar_numeric", "spatial": "spatial_tolerance", "spatial_numeric": "spatial_tolerance"}
                for claim_type in required_types:
                    required_handler = handler_for_type.get(str(claim_type), str(claim_type))
                    if required_handler not in declared_ids or not resolve_handler("deterministic_verifiers", required_handler):
                        failures.append("DETERMINISTIC_VERIFICATION_ROUTE_MISSING")
        for field, route_category, failure in (("valid_unenumerated_escalation", "valid_unenumerated", "VALID_UNENUMERATED_ESCALATION_MISSING"), ("uncertain_escalation", "uncertain", "UNCERTAIN_ESCALATION_MISSING")):
            route = contract.get(field)
            if isinstance(route, Mapping):
                available = route.get("available", True)
                route_id = route.get("route_id", route.get("handler_id"))
                if not route_id:
                    failures.append("EVALUATION_REPRESENTABILITY_CONTRACT_INVALID")
                prior = len(failures)
                if available is not True:
                    failures.append(failure if available is not True else "REPRESENTABILITY_HANDLER_UNRESOLVED")
                else:
                    validate_route("escalation_routes", route_id)
                    if len(failures) > prior and failures[-1] == "REPRESENTABILITY_HANDLER_NOT_PRODUCTION_BOUND":
                        pass
            elif route is not True:
                failures.append(failure)
            else:
                failures.append("EVALUATION_REPRESENTABILITY_CONTRACT_INVALID")
    # Flat contracts are retained for historical synthetic/unit fixtures.  No
    # default is used by the production SCQ path (allow_default=False).
    if not nested_contract:
        for key in REPRESENTABILITY_KEYS:
            item = contract.get(key)
            if key == "deterministic_handlers":
                if not isinstance(item, (list, tuple, set)) or not item:
                    failures.append("DETERMINISTIC_HANDLER_MISSING")
            elif item is not True:
                failures.append({
                    "o_schema_available": "O_SCHEMA_UNAVAILABLE",
                    "materializer_available": "G_OF_O_MATERIALIZER_MISSING",
                    "finding_extractor_available": "ATOMIC_FINDING_PATH_MISSING",
                    "semantic_handler_available": "SEMANTIC_HANDLER_MISSING",
                    "valid_unenumerated_escalation": "VALID_UNENUMERATED_ESCALATION_MISSING",
                    "uncertain_escalation": "UNCERTAIN_ESCALATION_MISSING",
                }[key])
    # Every explicitly required deterministic claim type must have a handler.
    deterministic = contract.get("deterministic_verification")
    if isinstance(deterministic, Mapping) and not nested_contract:
        handlers = {str(item) for item in contract.get("deterministic_handlers", ())}
        handler_for_type = {
            "exact": "exact",
            "scalar_numeric": "scalar_numeric",
            "spatial": "spatial_tolerance",
            "spatial_numeric": "spatial_tolerance",
        }
        for claim_type in deterministic.get("required_claim_types", ()) or ():
            required_handler = handler_for_type.get(str(claim_type), str(claim_type))
            if required_handler not in handlers:
                failures.append(f"DETERMINISTIC_HANDLER_MISSING:{claim_type}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failure_codes": list(dict.fromkeys(failures)),
        "contract": contract,
        "contract_present": contract_present,
        "resolved_routes": resolved_routes,
    }


__all__ = ["REPRESENTABILITY_KEYS", "DEFAULT_EVALUATION_REPRESENTABILITY_CONTRACT", "REPRESENTABILITY_HANDLER_REGISTRY", "get_handler_descriptor", "resolve_handler", "representability_handler_registry_snapshot", "validate_representability_contract"]
