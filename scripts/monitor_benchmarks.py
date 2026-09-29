#!/usr/bin/env python3
"""Read-only joint progress view. Never loads providers or starts workers."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


class ProgressWatch:
    """Read-only stagnation alarms; never restart tasks or alter judgments."""
    def __init__(self, stall_seconds=900):
        self.stall_seconds = stall_seconds
        self.progress = {}

    def observe(self, row, now=None):
        now = time.time() if now is None else now
        alarms = []
        if row.get('process') != 'RUNNING' or row.get('stop_requested'):
            self.progress.pop(row['experiment'], None)
            return alarms
        old = self.progress.setdefault(row['experiment'], {})
        metrics = {'answers': row['collection'].get('COMPLETED', 0),
                   'complete_scores': row['completed_answer_scores']}
        for name, count in metrics.items():
            previous, changed = old.get(name, (count, now))
            if count != previous:
                changed = now
            old[name] = count, changed
            waiting = (sum(row['collection'].get(s, 0) for s in ('PENDING', 'RUNNING')) > 0
                       if name == 'answers' else metrics['answers'] > metrics['complete_scores'])
            if waiting and now-changed >= self.stall_seconds:
                alarms.append(dict(kind='NO_NEW_'+name.upper(), seconds=round(now-changed),
                                   scheduling_state=row['grading']['scheduling_state']))
        return alarms


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def snapshot(output):
    state = read(output / 'collection_state.json')
    if not state:
        return {'experiment': output.name, 'process': 'NOT_PREPARED'}
    process = read(output / 'console_process.json')
    pid = process.get('pid')
    live = False
    if isinstance(pid, int):
        try:
            cmd = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
            live = any(Path(a.decode(errors='replace')).name in {'run_codex_benchmark.py', 'run_claude_benchmark.py'} for a in cmd if a)
        except OSError:
            pass
    report = read(output / 'reports/experiment_report.json')
    progress = read(output / 'judgment_progress.json')
    progress_current = progress.get('updated_epoch', 0) >= process.get('started_epoch', 0)
    health = read(output / 'api_health.json')
    shared = health.get('shared_http_admission') or {}
    shared_state = read(Path(shared['directory']) / 'state.json') if shared.get('directory') else {}
    recovery = read(output / 'infrastructure_recovery_queue.json')
    claude_health = read(output / 'claude_health.json')
    return dict(experiment=output.name, process='RUNNING' if live else 'STOPPED', pid=pid,
        exit_error=process.get('error_message'),
        stop_requested=(output / 'STOP_REQUESTED').exists(),
        collection=dict(Counter(s['status'] for s in state['slots'])),
        by_model={m:dict(Counter(s['status'] for s in state['slots'] if s['model_id']==m)) for m in state['configuration']['models']},
        scientifically_terminal=report.get('scientifically_terminal_slot_count', 0),
        completed_answer_scores=report.get('status_counts', {}).get('COMPLETED', 0),
        model_noncompletion=report.get('status_counts', {}).get('MODEL_NONCOMPLETION', 0),
        last_answer_epoch_by_model={m:max((s.get('finished_epoch', 0) for s in state['slots']
            if s['model_id'] == m and s['status'] == 'COMPLETED'), default=None) for m in state['configuration']['models']},
        recovery_blocked=len(recovery.get('current_blocked_slot_ids', [])),
        local_replay=read(output / 'replay_progress.json'),
        claude_provider_cooldown_seconds=max(0, round(claude_health.get('provider_cooldown_until', 0)-time.time())),
        grading=dict(running=progress.get('running', 0) if live and progress_current else 0, workers=progress.get('workers'),
                     current_invocation=progress_current,
                     resolved=progress.get('resolved', 0), atomic_dispatched=progress.get('atomic_requests_dispatched',0),
                     errors=progress.get('errors',0), scope=progress.get('scope'),
                     scheduling_state=(progress.get('scheduling_state') if live and progress_current
                                       else 'INITIALIZING' if live else 'STOPPED'),
                     retry_waiting=progress.get('retry_waiting', 0),
                     skipped_errors=len(progress.get('skipped_error_requests', [])),
                     deferred_execution=len(progress.get('deferred_execution_requests',[])),
                     update_age_seconds=round(time.time()-progress['updated_epoch']) if progress.get('updated_epoch') else None),
        api=dict(last_status=health.get('last_status'), wait_reason=health.get('wait_reason'),
                 cooldown_seconds=max(0,round(health.get('next_request_epoch',0)-time.time())),
                 shared_limit=shared.get('limit'),
                 shared_cooldown_seconds=max(0,round(shared_state.get('cooldown_until',0)-time.time())),
                 disabled=health.get('disabled',False), model_failures=health.get('model_failures',{})),
        report_path=str(output / 'reports/experiment_report.md'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=float, default=0, help='Refresh seconds; default one snapshot')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--stall-seconds', type=float, default=900,
                        help='Read-only alert after no new answer/full score; never restarts trials')
    args = parser.parse_args()
    if args.interval < 0:
        parser.error('interval cannot be negative')
    if args.stall_seconds <= 0:
        parser.error('stall-seconds must be positive')
    watch = ProgressWatch(args.stall_seconds)
    try:
        while True:
            rows = [snapshot(ROOT / 'outputs' / name) for name in ('expansion96_n3_subagents','expansion96_n3_claude')]
            for row in rows:
                row['stagnation_alerts'] = watch.observe(row)
            if args.interval and sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            if args.json:
                print(json.dumps(rows,ensure_ascii=False),flush=True)
            else:
                print(datetime.datetime.now().isoformat(timespec='seconds'))
                for r in rows:
                    print(f"\n{r['experiment']}  {r['process']}")
                    if 'collection' not in r:
                        continue
                    print('  答题:',r['collection'])
                    for model, counts in r['by_model'].items():
                        print('   ',model,counts)
                    g=r['grading']
                    print(f"  回答完整评分: {r['completed_answer_scores']}；模型未完成终态: {r['model_noncompletion']}；评分运行: {g['running']}/{g['workers']}；本轮子评分完成: {g['resolved']}；错误: {g['errors']}")
                    print(f"  调度: {g['scheduling_state']}；等待重试: {g['retry_waiting']}；错误隔离: {g['skipped_errors']}；补测阻塞: {r['recovery_blocked']}；Claude 供应商冷却: {r['claude_provider_cooldown_seconds']} 秒")
                    if not g['current_invocation']:
                        print('  子评分计数来自上次运行，本次尚未发布评分进度。')
                    if r['exit_error']:
                        print('  上次退出错误:', r['exit_error'])
                    if r['stagnation_alerts']:
                        print('  停滞提醒:', r['stagnation_alerts'])
                    for model, epoch in r['last_answer_epoch_by_model'].items():
                        print('  最后成功答题:', model, datetime.datetime.fromtimestamp(epoch).isoformat(timespec='seconds') if epoch else '无')
                    print(f"  待物化执行: {g['deferred_execution']}；进度更新时间距今: {g['update_age_seconds']} 秒；API: {r['api']}")
                    if r['local_replay']:
                        replay = r['local_replay']
                        print('  最近本地回放:', round(replay['elapsed_seconds'], 2), '秒；无模型调用的依赖处理:', replay.get('local_dependency_completion', {}))
                    if r['stop_requested']:
                        print('  已请求停止；下次显式启动自动续跑。')
                print(flush=True)
            if not args.interval:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()
