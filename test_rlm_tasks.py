"""Verify the trusted repository repair fixtures: python3 test_rlm_tasks.py."""
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile


def run(files, checks):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, source in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        script = "import sys\nsys.path.insert(0, " + repr(directory) + ")\n" + checks
        return subprocess.run(
            [sys.executable, "-I", "-c", script], cwd=directory,
            capture_output=True, text=True, timeout=10,
        )


def main():
    tasks = json.loads((Path(__file__).parent / "research/rlm_tasks.json").read_text(encoding="utf-8"))
    assert len(tasks) == 20
    assert len({task["id"] for task in tasks}) == len(tasks)
    repositories = {}
    counts = {"train": 0, "dev": 0, "holdout": 0}
    for task in tasks:
        assert task["split"] in counts
        counts[task["split"]] += 1
        previous = repositories.setdefault(task["repository"], task["split"])
        assert previous == task["split"], "A repository crosses splits"
        files = task["files"]
        assert 2 <= len(files) <= 5
        assert task["editable"] and len(set(task["editable"])) == len(task["editable"])
        assert set(task["editable"]) <= set(files)
        assert set(task["reference_patch"]) == set(task["editable"])
        for name, source in files.items():
            path = PurePosixPath(name)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in name
            assert str(path) == name and path.suffix == ".py"
            compile(source, name, "exec")
        compile(task["checks"], "checks.py", "exec")
        reference = {**files, **task["reference_patch"]}
        fixed = run(reference, task["checks"])
        assert fixed.returncode == 0, task["id"] + ": " + fixed.stderr
        broken = run(files, task["checks"])
        assert broken.returncode != 0, task["id"] + " has no detected defect"
        print(task["id"] + ": broken fixture fails; reference passes")
    assert counts == {"train": 10, "dev": 5, "holdout": 5}
    assert len(repositories) == 20
    print("20 repair fixtures verified; repository splits are disjoint.")


if __name__ == "__main__":
    main()
