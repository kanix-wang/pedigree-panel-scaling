"""Portable census and verification against independently recorded outcomes."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

BENCHMARKS = Path(__file__).resolve().parents[1]/'benchmarks'


def module(name):
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS/(name+'.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


census_module = module('census')
sys.modules.setdefault('census', census_module)
projection = module('project')
verification = module('verify')


class BenchmarkChecks(unittest.TestCase):
    def test_census_matches_every_recorded_nuclear_case(self):
        data = json.loads((BENCHMARKS/'reference.json').read_text())
        count = 0
        for key, row in data['cases'].items():
            if row['family'] != 'nuclear':
                continue
            expected = census_module.census(row['people'], row['genes'], row['profile'])
            self.assertEqual(expected, {k:row[k] for k in expected}, key)
            count += 1
        self.assertEqual(count, 12)

    def test_native_memory_matches_external_capacity_audit(self):
        row = projection.Projection().memory(10, 7, 'distinct')
        self.assertAlmostEqual(row['minimum_storage_GB'], 952.303464576, places=6)
        self.assertAlmostEqual(row['projected_peak_GB']['central']/2**30*1e9, 1103.118, places=2)

    def test_requested_thresholds_include_predecessor(self):
        p = projection.Projection()
        for axis, profile, expected in [('people','distinct',3), ('people','alternating',4),
                                        ('genes','distinct',7), ('genes','alternating',8)]:
            query = p.threshold(15,axis,profile,1000)
            for criterion, crossing in query['crossings'].items():
                self.assertEqual(crossing['first_dimension'], expected)
                before = crossing['preceding']
                if before:
                    value = before['minimum_storage_GB'] if criterion == 'minimum_storage_GB' else before['projected_peak_GB'][criterion]
                    self.assertLess(value, 1000)
                else:
                    self.assertEqual(axis, 'people')
                    self.assertEqual(expected, 3)

    def test_verifier_rejects_wrong_results_and_accepts_ties(self):
        row = json.loads((BENCHMARKS/'reference.json').read_text())['cases']['nuclear_n10_g4_distinct']
        changed = deepcopy(row)
        changed['action'] = 'M'
        self.assertEqual(verification.compare(changed,row), [])
        for key in ('states','value'):
            wrong = deepcopy(row)
            wrong[key] += 1
            self.assertTrue(verification.compare(wrong,row))
        wrong = deepcopy(row)
        wrong['root_action_values'].pop('M')
        self.assertTrue(verification.compare(wrong,row))


if __name__ == '__main__':
    unittest.main()
