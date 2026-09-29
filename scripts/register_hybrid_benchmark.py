"""Explicit audited amendment enabling third-party workers on existing slots."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_codex_benchmark as console
from scripts import collect_subagent_runs as collector
from flowintentbench.runtime_config import resolve_provider_configuration, resolve_api_key
from scripts.third_party_benchmark_transport import network_configuration


def register(output, config, amendment_reason=None):
    output, config = Path(output).resolve(), Path(config).resolve()
    with console.runner_lock(output):
        state = collector.read(output / 'collection_state.json')
        if any(s['status'] in {'RUNNING', 'FINISHED'} for s in state['slots']):
            raise RuntimeError('Drain or recover active trials before amending the protocol')
        collector.verify(state)
        preserved = {}
        for slot in state['slots']:
            if slot.get('run_record_sha256'):
                actual = collector.file_hash(output / slot['run_record_path'])
                if actual != slot['run_record_sha256']:
                    raise RuntimeError('Existing run hash mismatch')
                preserved[slot['slot_id']] = actual
        runtime = resolve_provider_configuration(config_path=config)
        resolve_api_key(runtime)  # Fail before changing anything if credentials are missing.
        if runtime.model_configuration.get('wire_api') != 'chat_completions':
            raise ValueError('Hybrid staged transport currently requires explicit chat_completions configuration')
        sources = {name:collector.file_hash(ROOT/'scripts'/name) for name in
            ('benchmark_hybrid.py','third_party_benchmark_transport.py','run_codex_file_judgments.py','run_codex_benchmark.py','report_subagent_experiment.py','recover_subagent_infrastructure.py',
             'codex_console_transport.py', 'grading_policy.py', 'batched_file_judgments.py',
             'lean_judgment_scheduler.py', 'carry_forward_file_judgments.py', 'console_evaluation_cache.py',
             'run_expansion_file_evaluation.py', 'replay_judgment_completion.py', 'api_judgment_recovery.py', 'shared_api_admission.py',
             '../flowintentbench/external_file_evaluator.py', '../flowintentbench/evaluator.py',
             '../flowintentbench/subagent_reporting.py', '../flowintentbench/metric_diagnostics.py',
             '../flowintentbench/deterministic_materialization.py', '../flowintentbench/expansion_recipe_materializer.py')}
        hybrid = dict(version='third-party-staged-hybrid-v1', api_runtime=runtime.without_secrets(),
            api_network=network_configuration(),
            sources=sources, models=list(collector.MODELS), reasoning_effort='xhigh',
            repetitions=3, timeout_seconds=collector.SOLVER_TIMEOUT_SECONDS,
            max_python_executions=collector.SOLVER_MAX_PYTHON_EXECUTIONS,
            mixed_agent_environments=True, claim_policy='OS lock plus durable unfinished-task marker',
            retry_policy='Explicit 429 only: bounded identical-request retry with Retry-After and jitter; ambiguous HTTP requests are never resent. Terminal infrastructure recovery uses a new audited attempt, capped at three unless a stopped-run per-trial authorization grants a bounded extension to six.',
            model_verification='Require returned model identifier equal requested model; third-party underlying weights cannot be independently attested',
            scientific_evaluator_unchanged=False,
            scoring_contract_unchanged=True,
            finding_routing_policy='qualitative-numeric-separation-v2',
            metric_diagnostics_policy='metric-validity-diagnostics-v1')
        from scripts.grading_policy import POLICY, POLICY_SHA256
        from scripts.recover_subagent_infrastructure import AUTO_RECOVERY_POLICY
        hybrid['infrastructure_recovery_policy'] = AUTO_RECOVERY_POLICY
        hybrid['reviewer_execution_policy'] = dict(POLICY, sha256=POLICY_SHA256,
            historical_responses='Preserved with original policy and reviewer provenance')
        hybrid['api_health_policy'] = 'Read-only model-catalog preflight before sampling; explicit network route; region restrictions disable API for the invocation; transient failures require healthy reprobe after cooldown; no automatic direct fallback'
        hybrid['cli_quota_policy'] = 'Explicit provider quota error disables CLI across restart; API continues. Only rejection before generation may use bounded audited replacement; possible generations are preserved for inspection.'
        hybrid['scheduling_policy'] = 'Per-invocation recorded solver roles; API-only solver mode reserves CLI for grading. Recorded total and per-model API solver caps; distinct pending trials only; eligible pending slots only; configurable stable recovery phases. Opt-in audited infrastructure requeue at startup and after terminal imports, before dependent replay. CLI-only scoring outages do not stop API collection.'
        hybrid['api_only_grading_policy'] = 'Zero CLI grading workers disables all CLI grader admission and login checks; one to eight API graders; wrapper defaults to two. Historical CLI judgments retained unchanged.'
        hybrid['rolling_grading_policy'] = 'The configured atomic limit bounds in-flight judgments; completed workers refill without waiting for slow wave siblings. Original per-answer/role batch contracts preserved.'
        hybrid['outage_recovery_policy'] = 'Target-model health admission before task reservation; independent idle health recovery; up to three cooled retries for known nonambiguous admission/rejection failures; unknown generations remain quarantined. Existing scientific responses preserved; changed O completion/evidence packets have new identities.'
        from scripts.api_judgment_recovery import POLICY as REVIEW_RECOVERY_POLICY
        hybrid['terminal_review_replacement_policy'] = REVIEW_RECOVERY_POLICY
        hybrid['shared_api_policy'] = 'Optional cross-process generation request cap, launch pacing and HTTP 429 cooldown keyed by endpoint/account; shared waits recorded as client queue time. Wrapper default four; Claude Code internal HTTP requests excluded.'
        hybrid['http_concurrency_policy'] = 'Recorded per schedule: one to eight in-flight requests (wrapper default three); shared endpoint pacing, Retry-After and endpoint/auth circuit. In concurrent mode transient transport/5xx failures require per-model cooldown and catalog reprobe without aborting the other model; successful concurrent calls cannot shorten an existing cooldown.'
        hybrid['efficiency_policy']={
            'raw_wall_time':'Preserved, never overwritten',
            'primary_efficiency':'Per transport, quality-qualified; exclude API trials with observed transport failures',
            'measured_overhead':'Client queue, pacing, Retry-After/circuit waits, failed HTTP durations recorded separately',
            'unknown_upstream_queue':'Cannot be separated; pure inference time remains null',
            'scientific_failure_policy':'API outages are infrastructure outcomes, not zero-scored model failures; only enabled bounded audited infrastructure replacements are queued. Completed/model-noncompletion observations are never resampled.',
            'wall_budget':'Solver budget is 720 seconds and 48 Python calls; known API failure exhausting a trial prevents model-failure attribution'}
        hp=output/'hybrid_protocol.json'
        if hp.exists():
            old=collector.read(hp)
            if any(old.get(k)!=v for k,v in hybrid.items()):
                if not amendment_reason or old['api_runtime']!=hybrid['api_runtime']:
                    raise RuntimeError('Existing hybrid protocol differs; explicit source amendment reason required; endpoint changes require separate review')
                previous=collector.file_hash(hp)
                history=output/'hybrid_protocol_versions';history.mkdir(exist_ok=True)
                archive_hybrid=history/(previous+'.json')
                if not archive_hybrid.exists():archive_hybrid.write_bytes(hp.read_bytes())
                hybrid['registered_epoch']=old['registered_epoch']
                hybrid['amendments']=[*old.get('amendments',[]),dict(epoch=time.time(),reason=amendment_reason,previous_sha256=previous)]
                collector.write(hp,hybrid)
        else:
            hybrid['registered_epoch']=time.time()
            collector.write(hp,hybrid)
        path=output/'console_protocol.json';old=path.read_bytes();digest=hashlib.sha256(old).hexdigest()
        archive=output/'console_protocol_versions'/(digest+'.json')
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(old)
        protocol=json.loads(old)
        protocol.update(transport='mixed_cli_and_third_party_api', project_api_calls='recorded_per_http_attempt',
                        direct_api_fallback='explicit_quota_rejection_only',
                        runner_sha256=sources['run_codex_benchmark.py'],
                        transport_sha256=sources['codex_console_transport.py'])
        protocol['scheduling_source_hashes'].update({name:sources[name] for name in
            ('run_codex_file_judgments.py','benchmark_hybrid.py','third_party_benchmark_transport.py',
             'grading_policy.py', 'batched_file_judgments.py', 'lean_judgment_scheduler.py', 'shared_api_admission.py')})
        protocol['scheduling_source_hashes']['carry_forward_file_judgments.py'] = sources['carry_forward_file_judgments.py']
        protocol['scheduling_source_hashes']['console_evaluation_cache.py'] = sources['console_evaluation_cache.py']
        protocol.setdefault('repair_history',[]).append(dict(epoch=time.time(),previous_protocol_sha256=digest,
            previous_protocol_path=str(archive),sources=sources,
            reason=amendment_reason or 'Explicit user authorization: third-party API and CLI jointly process remaining trials/judgments; rate-limit cooldown and bounded retries, distinct transport provenance, durable task claims.',
            preserved_run_count=len(preserved),hybrid_protocol_sha256=collector.file_hash(hp)))
        collector.write(path,protocol)
        audit=dict(epoch=time.time(),preserved_runs=preserved,configuration_sha256=state['configuration_sha256'],
                   previous_protocol_sha256=digest,protocol_sha256=collector.file_hash(path),sources=sources)
        collector.write(output/'preflight'/('hybrid_registration_'+str(time.time_ns())+'.json'),audit)
        console.prepare(output)
        return dict(preserved_runs=len(preserved),registered=True,config=str(config))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=console.DEFAULT_OUTPUT)
    parser.add_argument('--api-config',type=Path,default=ROOT/'config/yiapi.toml')
    parser.add_argument('--amendment-reason',help='Explicit reviewed source amendment while stopped; previous protocol bytes are archived')
    args=parser.parse_args()
    print(json.dumps(register(args.output_root,args.api_config,args.amendment_reason)))
