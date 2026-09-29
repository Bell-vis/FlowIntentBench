import json
import multiprocessing
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from scripts.shared_api_admission import SharedAPIAdmission
from scripts.monitor_benchmarks import snapshot


def hold_lane(root, ready, release):
    gate=SharedAPIAdmission('https://example.invalid','synthetic-key',limit=1,root=root,min_interval=0)
    with gate.admit(time.monotonic()+5):
        ready.set()
        release.wait(5)


def test_shared_limit_across_processes_and_release(tmp_path):
    context=multiprocessing.get_context('fork')
    ready,release=context.Event(),context.Event()
    process=context.Process(target=hold_lane,args=(tmp_path,ready,release))
    process.start()
    try:
        assert ready.wait(3)
        gate=SharedAPIAdmission('https://example.invalid','synthetic-key',limit=1,root=tmp_path,min_interval=0)
        with pytest.raises(TimeoutError):
            with gate.admit(time.monotonic()+.1):
                pytest.fail('Cross-process concurrency cap bypassed')
        release.set()
        process.join(3)
        assert process.exitcode==0
        with gate.admit(time.monotonic()+1):
            pass
        with pytest.raises(ValueError,match='limits differ'):
            SharedAPIAdmission('https://example.invalid','synthetic-key',limit=2,root=tmp_path,min_interval=0)
    finally:
        release.set()
        if process.is_alive():process.terminate()
        process.join()


def test_shared_cooldown_and_exception_release(tmp_path):
    gate=SharedAPIAdmission('https://example.invalid','synthetic-key',limit=1,root=tmp_path,min_interval=0)
    gate.cooldown(.2)
    other=SharedAPIAdmission('https://example.invalid','synthetic-key',limit=1,root=tmp_path,min_interval=0)
    with pytest.raises(TimeoutError):
        with other.admit(time.monotonic()+.05):pass
    with pytest.raises(RuntimeError):
        with gate.admit(time.monotonic()+1):raise RuntimeError('synthetic')
    with other.admit(time.monotonic()+1):pass
    assert 'synthetic-key' not in ''.join(p.read_text() for p in tmp_path.rglob('*') if p.is_file())


def test_monitor_does_not_create_experiments(tmp_path):
    assert snapshot(tmp_path/'missing')['process']=='NOT_PREPARED'
    assert not (tmp_path/'missing').exists()


def test_streaming_refills_before_slow_first_wave_finishes(tmp_path):
    from tests.test_lean_grading import requests, receipt, resolved
    from scripts.run_codex_file_judgments import run_judgments
    ex, inv, qs=requests(tmp_path,1,answer='first')
    all_requests=list(qs)
    for i in range(1,4):
        _,_,qs=requests(tmp_path,1,answer=str(i))
        all_requests.extend(qs)
    inv.write_text(json.dumps({'by_operation':{'semantic_match':[{'request_id':q['request_id']} for q in all_requests]}}))
    lock=threading.Lock(); calls=[]; third_started=threading.Event()
    def run(**kwargs):
        with lock:
            position=len(calls);calls.append(position)
        if position==0:
            assert third_started.wait(5),'Slow first wave blocked refill'
        if position==2:third_started.set()
        return receipt(resolved(),kwargs)
    result=run_judgments(ex,inv,tmp_path/'work','judge',limit=2,workers=2,lean=True,
                        runner=run,continuous=True)
    assert result['resolved']==4 and len(calls)==4


def test_api_only_controller_never_falls_back_to_cli(tmp_path,monkeypatch):
    from scripts import benchmark_hybrid as module
    monkeypatch.setattr(module,'ThirdPartyAgent',lambda *a:SimpleNamespace(runtime=SimpleNamespace(base_url='https://example.invalid'),key='test'))
    class Governor:
        def __init__(self,*a,**k):pass
        def configure_healthcheck(self,*a):pass
    monkeypatch.setattr(module,'APIGovernor',Governor)
    controller=module.HybridController(SimpleNamespace(),tmp_path,Path('unused'),cli_judges=0,
                                      api_judges=2,cli_solvers=False)
    controller.refresh_health=lambda:True
    seen=[]
    controller.api_judge_after_quota=lambda **kw:seen.append(kw) or {'completed':True}
    assert controller.judge_runner(model='gpt-6-astra')['completed']
    assert controller.cli_disabled and controller.api_judges==2 and len(seen)==1


def test_shared_admission_timeout_does_not_send_or_claim_unknown_response(tmp_path):
    from scripts.third_party_benchmark_transport import APIGovernor, APITransportFailure
    calls=[]
    governor=APIGovernor(tmp_path/'health.json',min_interval=0,opener=lambda *a,**k:calls.append(a))
    gate=SharedAPIAdmission('https://example.invalid','test',root=tmp_path/'shared',limit=1,min_interval=0)
    gate.cooldown(1)
    governor.shared_gate=gate
    send=governor.transport(tmp_path/'http','gpt-6-astra')
    with pytest.raises(APITransportFailure) as error:
        send('https://example.invalid',{},json.dumps({'model':'gpt-6-astra'}),.05)
    assert error.value.kind=='shared_admission_budget_exhausted'
    assert not error.value.ambiguous and not calls and governor.http_attempts==0
    assert send.telemetry['client_queue_wait_seconds'] >= .04
    assert not list((tmp_path/'http').glob('*/http_receipt.json'))


def test_http_429_updates_cross_process_cooldown(tmp_path):
    import urllib.error
    from scripts.third_party_benchmark_transport import APIGovernor, APITransportFailure
    def reject(*args,**kwargs):
        raise urllib.error.HTTPError('https://example.invalid',429,'rate limited',{'Retry-After':'37'},None)
    governor=APIGovernor(tmp_path/'health.json',min_interval=0,attempts=1,opener=reject)
    governor.shared_gate=SharedAPIAdmission('https://example.invalid','test',root=tmp_path/'shared',limit=1,min_interval=0)
    send=governor.transport(tmp_path/'http','gpt-6-astra')
    before=time.time()
    with pytest.raises(APITransportFailure):
        send('https://example.invalid',{},json.dumps({'model':'gpt-6-astra'}).encode(),1)
    state=json.loads((governor.shared_gate.directory/'state.json').read_text())
    assert state['cooldown_until'] >= before+37
    assert governor.http_attempts==1


def test_rollover_window_does_not_repeat_existing_responses(tmp_path):
    from tests.test_lean_grading import requests, receipt, resolved
    from scripts.run_codex_file_judgments import run_judgments
    ex,inv,_=requests(tmp_path,8)
    calls=[]
    def run(**kwargs):
        calls.append(1)
        return receipt({k:resolved() for k in kwargs['output_schema']['properties']},kwargs)
    first=run_judgments(ex,inv,tmp_path/'work','judge',limit=4,workers=2,lean=True,continuous=True,runner=run)
    assert first['resolved']==8 and len(calls)==2
    before={p.name:p.read_bytes() for p in (ex/'responses').glob('*.json')}
    run_judgments(ex,inv,tmp_path/'work','judge',limit=4,workers=2,lean=True,continuous=True,runner=run)
    assert len(calls)==2 and {p.name:p.read_bytes() for p in (ex/'responses').glob('*.json')}==before
