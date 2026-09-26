"""Explicit identity-only repair; preserve old evidence and measurement code.

Only reference_binaries may differ. Inputs, admission, capture boundaries and
timing remain AST-identical. This is not a general harness-revision bypass.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def digest(data):
    return hashlib.sha256(data).hexdigest()


def measurement_ast(source):
    tree = ast.parse(source)
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == "reference_binaries"]
    if len(found) != 1:
        raise ValueError("exactly one identity collector required")
    tree.body.remove(found[0])
    return ast.dump(tree, include_attributes=False)


def prove_identity_only(before, after):
    if measurement_ast(before) != measurement_ast(after):
        raise ValueError("measurement changed outside reference_binaries")


def cell_key(row):
    return row["workload"], row["gate"], row["family"]


def repair(root, repo, current):
    path = root / "matrix.json"
    state = json.loads(path.read_text())
    if state.get("identity_epochs"):
        raise ValueError("identity repair already registered")
    old = subprocess.check_output(["git", "show", "71534d6:benchmarks/profile_sm90_libraries.py"], cwd=repo)
    new = (repo / "benchmarks/profile_sm90_libraries.py").read_bytes()
    if digest(old) != state["bindings"]["harness"] or digest(new) != current["harness"]:
        raise ValueError("repair does not bind the actual previous/current harness")
    if {k: v for k, v in state["bindings"].items() if k != "harness"} != {
            k: v for k, v in current.items() if k != "harness"}:
        raise ValueError("repair may not change inputs or binaries")
    prove_identity_only(old, new)
    expected = {("batch2", -.1, "flashqla"), ("seq512", -.1, "flashqla"),
                ("batch4", -.1, "flashqla"), ("heads64-gva4", -.1, "flashqla")}
    retries = []
    for row in state["cells"]:
        if row["family"] != "flashqla" or row["status"] not in ("PASS", "FAIL"):
            continue
        log = (root / row["directory"] / "run.log").read_text()
        fresh = "TileLang begins to compile kernel" in log
        if row["status"] == "FAIL":
            if "reference JIT binary identity unavailable" not in log:
                raise ValueError("cannot retry an unrelated failure as identity-only")
            retries.append(row)
        elif fresh:
            retries.append(row)
    if {cell_key(row) for row in retries} != expected:
        raise ValueError("identity-repair attempt inventory differs from audited four cells")
    backup = root / "matrix-before-identity-repair.json"
    snapshots = root / "identity-epochs"
    if backup.exists() or snapshots.exists():
        raise ValueError("refuse to overwrite identity history")
    backup.write_bytes(path.read_bytes())
    snapshots.mkdir()
    epochs = []
    collector = (repo / "benchmarks/sm90_jit_identity.py").read_bytes()
    for name, source in (("mapped-only", old), ("live-module", new)):
        snapshot = snapshots / f"{name}.py"
        snapshot.write_bytes(source)
        epoch = dict(harness=digest(source), snapshot=str(snapshot.relative_to(root)))
        if name == "live-module":
            helper = snapshots / "live-module-collector.py"
            helper.write_bytes(collector)
            epoch.update(collector=digest(collector), collector_snapshot=str(helper.relative_to(root)))
        epochs.append(epoch)
    for row in state["cells"]:
        if row["status"] == "PASS":
            row["measurement_epoch"] = digest(old)
    for row in retries:
        previous = dict(row)
        row.clear()
        row.update(workload=previous["workload"], gate=previous["gate"], family=previous["family"],
                   directory=previous["directory"] + "-identity-r2", status="PENDING",
                   prior_attempts=[dict(reason="INCOMPLETE_REFERENCE_BINARY_IDENTITY", record=previous)])
    state.update(bindings=current, identity_epochs=epochs,
                 identity_repair=dict(utc=datetime.now(timezone.utc).isoformat(),
                     proof="AST_IDENTICAL_EXCEPT_REFERENCE_BINARIES", retries=sorted(expected),
                     previous_matrix_sha256=digest(backup.read_bytes())))
    state["status_counts"] = {s: sum(r["status"] == s for r in state["cells"])
                              for s in ("PENDING", "RUNNING", "PASS", "FAIL")}
    new_path = path.with_suffix(".json.new")
    new_path.write_text(json.dumps(state, indent=2) + "\n")
    new_path.replace(path)


def epochs(root, state):
    entries = state.get("identity_epochs")
    if not entries:
        return {state["bindings"]["harness"]: {"harness": state["bindings"]["harness"]}}
    if len(entries) != 2:
        raise ValueError("unexpected identity epoch denominator")
    source = []
    for entry in entries:
        data = (root / entry["snapshot"]).read_bytes()
        if digest(data) != entry["harness"]:
            raise ValueError("changed harness snapshot")
        source.append(data)
        if "collector" in entry:
            if digest((root / entry["collector_snapshot"]).read_bytes()) != entry["collector"]:
                raise ValueError("changed collector snapshot")
    prove_identity_only(*source)
    if entries[-1]["harness"] != state["bindings"]["harness"]:
        raise ValueError("current harness is not the repaired epoch")
    if digest((root / "matrix-before-identity-repair.json").read_bytes()) != state["identity_repair"]["previous_matrix_sha256"]:
        raise ValueError("changed pre-repair evidence")
    return {entry["harness"]: entry for entry in entries}
