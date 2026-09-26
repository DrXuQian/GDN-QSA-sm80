import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/"tools/analyze_sm90_roles.py"
spec = importlib.util.spec_from_file_location("roles",path)
roles = importlib.util.module_from_spec(spec); spec.loader.exec_module(roles)


class RoleTrace(unittest.TestCase):
    def fixture(self):
        data = [0]*(64*64*3*16)
        for cta in range(2):
            for chunk in range(3):
                for role,n in enumerate(roles.POINTS):
                    for point in range(n):
                        data[roles.index(cta,chunk,role,point)] = 100000+10000*chunk+100*role+point*10
        return data

    def test_fixed_owner_denominator(self):
        self.assertEqual(roles.analyze(self.fixture(),2,3)["expected_stamps"],210)

    def test_missing_stamp_and_omitted_cta_are_red(self):
        bad=self.fixture(); bad[roles.index(1,2,2,12)]=0
        with self.assertRaises(AssertionError): roles.analyze(bad,2,3)
        with self.assertRaises(AssertionError): roles.analyze(self.fixture(),1,3)

    def test_reordered_or_unowned_stamp_is_red(self):
        bad=self.fixture(); bad[roles.index(0,0,1,7)]=1
        with self.assertRaises(AssertionError): roles.analyze(bad,2,3)
        bad=self.fixture(); bad[roles.index(0,0,0,15)]=999
        with self.assertRaises(AssertionError): roles.analyze(bad,2,3)


if __name__=="__main__": unittest.main()
