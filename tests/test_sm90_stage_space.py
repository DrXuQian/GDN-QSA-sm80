import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from sm90_stage_space import space, validate, admit_types
from build_sm90_stage_sweep import native_contract


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

    def test_completion_defect_is_not_a_resource_skip(self):
        # Synthetic rows test checker behavior; actual SASS is checked separately.
        body=['SYNCS.EXCH.64 R0, [R1], R2']*32+[
            'HGMMA.64x64x16.F32.BF16 R0, R1, R2',
            'WARPGROUP.DEPBAR.LE gsb0, 0x0']
        parent={(g,h):body for g in ('float','cutlass::bfloat16_t') for h in ('false','true')}
        native_contract(parent,parent,space()[0])
        key=next(iter(parent))
        lost=dict(parent);lost[key]=body[:-1]
        relaxed=dict(parent);relaxed[key]=body[:-1]+['WARPGROUP.DEPBAR.LE gsb0, 0x1']
        missing=dict(parent);missing.pop(key)
        for plant in (lost,relaxed,missing):
            with self.assertRaises(AssertionError):native_contract(plant,parent,space()[0])


if __name__ == '__main__': unittest.main()
