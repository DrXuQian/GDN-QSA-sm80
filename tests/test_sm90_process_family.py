import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from sm90_process_family import family_ids


class Family(unittest.TestCase):
    def test_descendants_and_namespaces(self):
        p={1:dict(ppid=0,nspid=[1,100]),2:dict(ppid=1,nspid=[2,101]),
           3:dict(ppid=2,nspid=[3,102]),4:dict(ppid=0,nspid=[4,103])}
        self.assertEqual(family_ids(p,1),{1,100,2,101,3,102})
        self.assertNotIn(103,family_ids(p,1))

    def test_no_stale_pid_permission(self):
        before={1:dict(ppid=0,nspid=[1]),2:dict(ppid=1,nspid=[2])}
        self.assertIn(2,family_ids(before,1))
        self.assertNotIn(2,family_ids({1:before[1],2:dict(ppid=0,nspid=[2])},1))
        self.assertNotIn(2,family_ids({1:before[1]},1))

    def test_missing_root_is_error(self):
        with self.assertRaises(RuntimeError):family_ids({},1)


if __name__=='__main__':unittest.main()
