"""Compile actual kernel count expressions; no parallel host scheduler model."""
from pathlib import Path
import re
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=(ROOT/'csrc/backends/sm90/cula/kda/sm90/kernel/kernel_kda_fwd.hpp').read_text()


def compile_counts(plant=False):
    expressions={name:re.search(name+r'_pipeline_params.consumer_arv_count\s*=([^;]+);',SOURCE)[1]
                 for name in ('alpha','beta')}
    if plant:
        expressions['alpha']=expressions['alpha'].replace('? 0 :','? 32 :')
    # The RHS is copied verbatim from the actual launcher and compiled by C++.
    # Expected participants are independently enumerated: state WGs256, aux128,
    # plus the legacy extractor warp32 only when that consumer still exists.
    cpp='''namespace cutlass { constexpr int NumThreadsPerWarp=32; }
template<bool Owned> struct Case {
 struct CollectiveMainloop { static constexpr bool ScalarAuxOwnsMetadata=Owned; };
 static constexpr int NumStateMathThreads=256, NumAuxMathThreads=128;
 static constexpr int alpha='''+expressions['alpha']+''';
 static constexpr int beta='''+expressions['beta']+''';
};
static_assert(Case<false>::alpha==416, "legacy alpha consumers");
static_assert(Case<false>::beta==384, "legacy beta consumers");
static_assert(Case<true>::alpha==384, "owned alpha consumers");
static_assert(Case<true>::beta==128, "owned beta consumers");
'''
    return subprocess.run(['c++','-std=c++17','-x','c++','-fsyntax-only','-'],
                          input=cpp,text=True,capture_output=True)


class Counts(unittest.TestCase):
    def test_actual_expressions_both_ownership_states(self):
        result=compile_counts()
        self.assertEqual(result.returncode,0,result.stderr)
    def test_retaining_dead_consumer_is_red(self):
        result=compile_counts(True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('static assertion failed: owned alpha consumers',result.stderr)

if __name__=='__main__':unittest.main()
