import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from sm90_stage_space import space, validate, admit_types


class StageSpace(unittest.TestCase):
    def test_denominator_is_cartesian_not_only_unique_ids(self):
        rows = space()
        self.assertEqual(len(rows), 32)
        for plant in (rows[:-1], rows+[rows[0]], rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError): validate(plant)
        wrong = copy.deepcopy(rows)
        wrong[31]['stages'] = wrong[30]['stages']
        with self.assertRaises(ValueError): validate(wrong)

    def test_actual_types_not_requested_label(self):
        for cell in space():
            actual = [dict(config=cell['id'],gate_fp32=g,initial=h,
                           stages=cell['stages'],shared_bytes=100)
                      for g in (0,1) for h in (0,1)]
            admit_types(cell,actual)
            with self.assertRaises(ValueError): admit_types(cell,actual[:-1])
            bad=copy.deepcopy(actual);bad[0]['stages'][1] += 1
            # deepcopy preserves aliasing inside actual; the cell remains intact.
            with self.assertRaises(ValueError): admit_types(cell,bad)


if __name__ == '__main__': unittest.main()
