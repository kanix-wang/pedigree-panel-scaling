"""Project nuclear-family resources from measured runs, without optimization.

GB means 10**9 bytes. Counts are conditional exact integers; memory and time
are empirical extrapolations, not confidence intervals or machine ETAs.
"""
import argparse
from bisect import bisect_left
import json
import math
from pathlib import Path
from statistics import median

from census import census, key_bits, state_count

HERE = Path(__file__).resolve().parent


def dictionary_bytes(states):
    capacity = 8
    while capacity*2//3 < states:
        capacity *= 2
    width = 1 if capacity <= 128 else 2 if capacity <= 32768 else 4 if capacity <= 2**31 else 8
    return 32+capacity*width+24*(capacity*2//3)


def envelope(values):
    return dict(low=min(values), central=median(values), high=max(values))


def prime(n):
    for p in (2,3,5,7,11,13,17,19,23,29,31,37):
        if n % p == 0:
            return n == p
    d, s = n-1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2,325,9375,28178,450775,9780504,1795265022):
        if a % n == 0:
            continue
        x = pow(a,d,n)
        if x in (1,n-1):
            continue
        for _ in range(s-1):
            x = x*x % n
            if x == n-1:
                break
        else:
            return False
    return True


class Projection:
    def __init__(self):
        self.data = json.loads((HERE/'reference.json').read_text())
        self.cases = self.data['cases']
        self.buckets = self.data['native_bucket_sequence']
        self.base = median(r['initial_rss_bytes'] for r in self.cases.values())
        native = [r for r in self.data['memory_calibration_epochs'] if r['family'] == 'nuclear']
        self.native_rates = envelope([(r['end_rss_bytes']-r['start_rss_bytes'])/(r['end_states']-r['start_states']) for r in native])
        self.residuals = envelope([r['result_stage_rss_bytes']-r['initial_rss_bytes']
                                  -self.native_rates['central']*r['states']-8*self.capacity(r['states'])
                                  for r in self.cases.values() if r['family'] == 'nuclear'])
        calibration_path = HERE/'python_calibration.json'
        self.python_runs = json.loads(calibration_path.read_text())['results'] if calibration_path.exists() else []
        for r in self.python_runs:
            ref = self.cases[r['case']]
            fields = ('model', 'people', 'genes', 'profile', 'states', 'transitions',
                      'terminal_actions_closed', 'independent_states_closed')
            if (not r['verification_passed'] or r['backend'] != 'python' or r['family'] != 'nuclear'
                    or r['process_exit_code'] != 0 or any(r[k] != ref[k] for k in fields)
                    or not math.isfinite(r['solver_seconds']) or r['solver_seconds'] <= 0
                    or not math.isfinite(r['peak_rss_bytes']) or r['peak_rss_bytes'] <= 0
                    or not math.isclose(r['value'], ref['value'], abs_tol=1e-10, rel_tol=1e-10)
                    or r['root_action_values'].keys() != ref['root_action_values'].keys()
                    or any(not math.isclose(v, ref['root_action_values'][a], abs_tol=1e-10, rel_tol=1e-10)
                           for a,v in r['root_action_values'].items())):
                raise ValueError('Invalid Python calibration')

    def capacity(self, states):
        while self.buckets[-1] < states:
            candidate = 2*self.buckets[-1]+1
            while not prime(candidate):
                candidate += 2
            self.buckets.append(candidate)
        index = bisect_left(self.buckets, states)
        return self.buckets[index]

    def memory(self, people, genes, profile):
        states, bits = state_count(people, genes, profile), key_bits(people, genes)
        native = bits <= 64
        # Captured 64-bit CPython layout: tuple header40, pointer8, pair56, entry24.
        key_size = 40+8*(genes+2)
        minimum = states*(48 if native else key_size+56+24)
        if native:
            final = {k: self.base+v*states+8*self.capacity(states)+self.residuals[k]
                     for k,v in self.native_rates.items()}
            peaks = dict(final)
            for old,new in zip(self.buckets, self.buckets[1:]):
                if old+1 > states:
                    break
                peaks['high'] = max(peaks['high'], self.base+self.native_rates['high']*(old+1)
                                    +8*(old+new)+self.residuals['high'])
            method = 'Measured native state-growth slopes, bucket capacities, and allocation-overlap scenario'
        elif self.python_runs:
            # Calibrate non-key object/allocator demand after removing the one
            # dictionary and fixed-length tuple keys; then resize for target G.
            rates = envelope([(r['peak_rss_bytes']-self.base-dictionary_bytes(r['states']))/r['states']
                              -(40+8*(r['genes']+2)) for r in self.python_runs])
            table = dictionary_bytes(states)
            final = {k: max(minimum, self.base+(v+key_size)*states+table) for k,v in rates.items()}
            peaks = dict(final)
            # Retention of older dictionary allocations is uncertain. Their
            # geometric total is bounded here by one additional current table.
            peaks['high'] += table
            method = 'Two measured Python runs; tuple-size adjustment, dictionary capacity, and prior-table retention scenario'
        else:
            final = peaks = None
            method = 'No Python peak calibration; only necessary storage is available'
        return dict(people=people, genes=genes, profile=profile, states=states, key_bits=bits,
                    backend='native' if native else 'python', minimum_storage_GB=minimum/1e9,
                    projected_peak_GB=None if peaks is None else {k:v/1e9 for k,v in peaks.items()},
                    memory_method=method)

    def project(self, people, genes, profile):
        result = self.memory(people, genes, profile)
        counts = census(people, genes, profile)
        result.update(counts)
        if result['backend'] == 'native':
            candidates = [dict(r, case=k) for k,r in self.cases.items()
                          if r['family']=='nuclear' and r['profile']==profile]
        else:
            candidates = [r for r in self.python_runs if r['profile']==profile]
        result['runtime'] = None
        if candidates:
            reference = max(candidates, key=lambda r: (r['people']==people, r['genes']==genes, r['states']))
            hours = reference['solver_seconds']/reference['transitions']*counts['transitions']/3600
            sensitivity = [r['solver_seconds']/r['transitions']*counts['transitions']/3600 for r in candidates]
            result['runtime'] = dict(reference_case=reference['case'], reference_people=reference['people'],
                                     reference_genes=reference['genes'], reference_solver_seconds=reference['solver_seconds'],
                                     reference_branches=reference['transitions'], compute_anchor_hours=hours,
                                     compute_anchor_days=hours/24, compute_anchor_years=hours/(24*365.25),
                                     observed_throughput_sensitivity_hours=[min(sensitivity),max(sensitivity)] if len(candidates)>1 else None,
                                     matched_fixed_people=reference['people']==people,
                                     matched_fixed_genes=reference['genes']==genes,
                                     interpretation='Whole-case branch-count transfer; unvalidated at target size, excludes paging, not an ETA')
        result['qualification'] = ('Conditional current-representation census. Projections assume benchmark allocator/throughput; '
                                   'no guarantee of fit, completion time, optimality at an unrun size, or real-world parameter applicability.')
        return result

    def threshold(self, fixed, axis, profile, gb):
        # A capacity search over integer counts, not a solver stopping limit.
        criteria = ('minimum_storage_GB', 'central', 'high')
        found, previous = {}, None
        dimension = 3 if axis == 'people' else 1
        while len(found) < len(criteria):
            people, genes = (dimension, fixed) if axis == 'people' else (fixed, dimension)
            row = self.memory(people, genes, profile)
            for criterion in criteria:
                value = row['minimum_storage_GB'] if criterion == 'minimum_storage_GB' else (
                    row['projected_peak_GB'][criterion] if row['projected_peak_GB'] else None)
                if value is not None and criterion not in found and value >= gb:
                    found[criterion] = dict(first_dimension=dimension, at=self.project(people, genes, profile), preceding=previous)
            if row['projected_peak_GB'] is None and 'minimum_storage_GB' in found:
                break
            previous = row
            dimension += 1
        return dict(threshold_GB=gb, varied_axis=axis, fixed_genes=fixed if axis=='people' else None,
                    fixed_people=fixed if axis=='genes' else None, profile=profile, crossings=found,
                    criterion_note='First integer >= threshold under each stated criterion, not first observed memory failure')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--people', type=int)
    parser.add_argument('--genes', type=int)
    parser.add_argument('--fixed-people', type=int)
    parser.add_argument('--fixed-genes', type=int)
    parser.add_argument('--threshold-gb', type=float, default=1000)
    parser.add_argument('--profile', choices=('distinct','alternating'), default='distinct')
    args = parser.parse_args()
    p = Projection()
    if not math.isfinite(args.threshold_gb) or args.threshold_gb <= 0:
        parser.error('Threshold must be positive')
    if args.fixed_people is not None or args.fixed_genes is not None:
        if (args.fixed_people is not None and args.fixed_genes is not None) or args.people is not None or args.genes is not None:
            parser.error('Choose exactly one fixed-axis query or a people/genes query')
        if (args.fixed_people is not None and args.fixed_people < 3) or (args.fixed_genes is not None and args.fixed_genes < 1):
            parser.error('Require at least three people and one gene')
        result = p.threshold(args.fixed_people or args.fixed_genes,
                             'genes' if args.fixed_people is not None else 'people', args.profile, args.threshold_gb)
    elif args.people is not None and args.genes is not None and args.people >= 3 and args.genes >= 1:
        result = p.project(args.people, args.genes, args.profile)
    else:
        parser.error('Supply --people N --genes G, or --fixed-people N, or --fixed-genes G')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
