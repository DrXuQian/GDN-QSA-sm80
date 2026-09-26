"""Admission-only child-process ownership; never allow unrelated GPU jobs."""
import os
from pathlib import Path


def family_ids(snapshot, root):
    if root not in snapshot:
        raise RuntimeError('monitor root vanished')
    owned={root}
    while True:
        following=owned | {pid for pid,row in snapshot.items() if row['ppid'] in owned}
        if following==owned:break
        owned=following
    return {value for pid in owned for value in snapshot[pid]['nspid']}


def snapshot_processes():
    result={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():continue
        try:
            # Fields after the final ')' start at stat field3. Never split
            # command names on spaces and accidentally parse a false parent.
            fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
            status=(entry/'status').read_text().splitlines()
            nspid=next((line.split()[1:] for line in status if line.startswith('NSpid:')), [entry.name])
            result[int(entry.name)]=dict(ppid=int(fields[1]),starttime=int(fields[19]),nspid=list(map(int,nspid)))
        except (FileNotFoundError,ProcessLookupError):
            continue
    return result


def admission_watch(base):
    class FamilyWatch(base):
        def sample(self,idle=False):
            snapshot=snapshot_processes()
            # Recompute for each observation; do not keep exited child PIDs
            # in a whitelist that could accept a later unrelated reuse.
            self.allowed=family_ids(snapshot,os.getpid())
            super().sample(idle=idle)
            self.records[-1]['allowed_process_family']=sorted(self.allowed)
    return FamilyWatch
