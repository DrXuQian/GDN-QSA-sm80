import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('intervals',Path(__file__).resolve().parents[1]/'dev/backends/sm90_sass_intervals.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
FIXTURE=''' .type first,@function
first:
 /*0000*/ HGMMA.64x64x16.F32.BF16 R0, R4, R8 ;
 //## File "order.cuh", line 4
 /*0010*/ LDL R3, [R1] ;
 /*0020*/ @P0 BRA label ;
 .type second,@function
second:
 /*0000*/ STL [R1], R3 ;
'''

class Intervals(unittest.TestCase):
    def test_symbol_and_lowercase_mma_shape(self):
        result=module.inspect(FIXTURE,'first',0,32)
        self.assertEqual(result['counts'],dict(BRA=1,HGMMA=1,LDL=1))
        self.assertEqual(result['local_memory'][0]['source'],dict(file='order.cuh',line=4))
    def test_missing_symbol_endpoint_and_duplicate_pc_are_red(self):
        for source,symbol,end in ((FIXTURE,'fir',32),(FIXTURE,'first',48),
                                  (FIXTURE.replace('/*0010*/','/*0000*/'),'first',32)):
            with self.assertRaises(ValueError):module.inspect(source,symbol,0,end)

if __name__=='__main__':unittest.main()
