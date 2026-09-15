"""Rerun recorded exact cases in fresh processes; record time and peak memory.

No solver limits or approximation. Verification failures exit nonzero.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
COUNTERS = ('states', 'transitions', 'terminal_actions_closed', 'independent_states_closed')


def peak_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1024 if sys.platform.startswith('linux') else 1)) if sys.platform == 'darwin' or sys.platform.startswith('linux') else None
    except ImportError:
        return None


def compare(actual, reference, tolerance=1e-10):
    failures = []
    for field in COUNTERS:
        if actual[field] != reference[field]:
            failures.append(field)
    close = lambda a, b: math.isclose(a, b, abs_tol=tolerance, rel_tol=tolerance)
    if not close(actual['value'], reference['value']):
        failures.append('optimal value')
    a, b = actual['root_action_values'], reference['root_action_values']
    if a.keys() != b.keys() or any(not close(a[k], b[k]) for k in a.keys() & b.keys()):
        failures.append('root action values')
    if actual['action'] not in b or not close(b[actual['action']], reference['value']):
        failures.append('chosen-action value (ties allowed)')
    return failures


def worker(case_id, backend, output):
    import pedigree_panel_scaling as package
    from pedigree_panel_scaling import load_case, make_inference, solve_exact
    data = json.loads((HERE/'reference.json').read_text())
    reference = data['cases'][case_id]
    with tempfile.TemporaryDirectory() as directory:
        case_path = Path(directory)/'case.json'
        case_path.write_text(json.dumps(reference['model']))
        case = load_case(case_path)
    engine = make_inference(case)
    last = -15.0

    def progress(p):
        nonlocal last
        if p.elapsed_seconds-last >= 15:
            last = p.elapsed_seconds
            print(json.dumps(dict(case=case_id, status='running', solver_seconds=p.elapsed_seconds,
                                  states=p.states_evaluated, transitions=p.transitions_evaluated,
                                  peak_rss_bytes_so_far=peak_bytes())), file=sys.stderr, flush=True)

    if backend == 'python' and reference['family'] == 'nuclear':
        from pedigree_panel_scaling._nuclear_exact import _solve_python
        solution = _solve_python(engine, progress=progress)
    else:
        solution = solve_exact(engine, progress=progress)
    name = lambda action: 'STOP' if action < 0 else case.people[action]
    storage = type(getattr(solution._codec, 'memo', None)).__name__
    module_dir = Path(package.__file__).parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for pattern in ('*.py', '*.cpp', '*.so', '*.pyd') for p in module_dir.glob(pattern)}
    actual = dict(case=case_id, status='optimal', backend='native' if storage == '_PackedMemo' else 'python',
                  backend_storage=storage, family=reference['family'], people=reference['people'], genes=reference['genes'],
                  profile=reference['profile'], model=reference['model'], value=solution.root_value,
                  action=name(solution.root_action), root_action_values={name(a): v for a,v in solution.root_action_values.items()},
                  states=solution.states_evaluated, transitions=solution.transitions_evaluated,
                  terminal_actions_closed=solution.terminal_actions_closed, independent_states_closed=solution.independent_states_closed,
                  solver_seconds=solution.elapsed_seconds, cpu_seconds=time.process_time(), peak_rss_bytes=peak_bytes(),
                  python=platform.python_version(), platform=platform.platform(), source_sha256=hashes,
                  limits=dict(max_states=None, max_transitions=None, max_seconds=None),
                  result_available_utc=datetime.now(timezone.utc).isoformat(),
                  reference_result_sha256=reference['source_result_sha256'])
    actual['verification_failures'] = compare(actual, reference)
    actual['verification_passed'] = not actual['verification_failures']
    # Binary hashes can differ across builds. Source equality is separate evidence.
    actual['python_sources_match_reference'] = all(hashes.get(n) == h for n,h in data['source_sha256'].items() if n.endswith('.py'))
    actual['reference_backend_matches'] = storage == reference['backend_storage']
    Path(output).write_text(json.dumps(actual, indent=2)+'\n')
    return 0 if actual['verification_passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_mutually_exclusive_group()
    select.add_argument('--case', action='append', help='Recorded case ID; may be repeated')
    select.add_argument('--all', action='store_true', help='Rerun all 18 cases sequentially; takes many hours')
    select.add_argument('--list', action='store_true')
    parser.add_argument('--backend', choices=('auto', 'python'), default='auto')
    parser.add_argument('--output', type=Path, help='New JSON report file; existing files are never overwritten')
    parser.add_argument('--worker-output', help=argparse.SUPPRESS)
    args = parser.parse_args()
    data = json.loads((HERE/'reference.json').read_text())
    if args.list:
        for key, r in data['cases'].items():
            print(f"{key}: {r['solver_seconds']:.3f} s, {r['peak_rss_bytes']/1e9:.3f} GB observed")
        return 0
    selected = list(data['cases']) if args.all else args.case or ['nuclear_n10_g4_distinct']
    for key in selected:
        if key not in data['cases']:
            parser.error(f'Unknown recorded case {key!r}; use --list')
    if args.worker_output:
        return worker(selected[0], args.backend, args.worker_output)
    if args.output and args.output.exists():
        parser.error('Output already exists; choose a new filename')
    results = []
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    with tempfile.TemporaryDirectory() as directory:
        for index, key in enumerate(selected):
            result_path = Path(directory)/f'{index}.json'
            started = time.perf_counter()
            completed = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--case', key,
                                        '--backend', args.backend, '--worker-output', str(result_path)], env=env)
            row = json.loads(result_path.read_text()) if result_path.exists() else dict(case=key, status='failed', verification_passed=False)
            row.update(process_exit_code=completed.returncode, process_wall_seconds=time.perf_counter()-started)
            row['verification_passed'] = bool(row['verification_passed'] and completed.returncode == 0)
            results.append(row)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  reference_sha256=hashlib.sha256((HERE/'reference.json').read_bytes()).hexdigest(),
                  all_passed=all(r['verification_passed'] for r in results), results=results,
                  timing_note='Solver time ends at result availability. Process wall time includes startup and natural shutdown.',
                  memory_note='Peak RSS through result-stage sampling; bytes on macOS/Linux, unavailable (null) on unsupported platforms. Not a guaranteed whole-lifetime peak.',
                  performance_note='Time and memory are measured, not pass/fail thresholds; the historical machine and backend can differ.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
