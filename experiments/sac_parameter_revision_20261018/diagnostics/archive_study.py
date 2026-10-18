"""Archive completed profit-only SAC studies without distributing checkpoints.

Run from the repository with:
    python -B checks/sac_revision/archive_study.py --dry-run
    python -B checks/sac_revision/archive_study.py

The default destination is experiments/sac_parameter_revision_20261018.
Repeated runs update managed files, add newly completed runs, and preserve the
manually written README and other unmanaged files. Sources are never modified.
No model is loaded, and no training or evaluation is executed.

Probe scripts are copied byte-for-byte. Their original checks/sac_revision
location may be required to run them; copying does not rewrite their ROOT logic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


STUDY_NAME = "sac_parameter_revision_20261018"
EVALUATION_PATTERN = re.compile(r"evaluation_(\d+)\.json\Z")
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".pdf", ".xml", ".zip", ".gz", ".tar", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".tif", ".tiff",
    ".pyc", ".pyo",
}
LITERATURE_NAMES = (
    "parameter_evidence.json", "baseline_parameter_ratios.json",
    "penetration_sampling.json", "参数文献核对.md",
    "renewable_share_supplement.json",
)
DIAGNOSTIC_STEMS = (
    "response_parameter_probe", "response_scenario_probe", "exact_response_elasticity",
    "build_response_diagnostic_summary",
)
DIAGNOSTIC_NAMES = {
    stem + suffix for stem in DIAGNOSTIC_STEMS for suffix in (".py", ".json")
} | {"response_diagnostic_summary.json", "archive_study.py"}
FINAL_ANALYSIS_JSON_NAMES = (
    "final_evaluation_plan.json", "final_500_checkpoints.json", "final_response_effects.json",
)
FINAL_ANALYSIS_DIRECTORIES = ("final_candidate_analysis", "final_baseline_analysis")


def default_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "train_sac.py").is_file() and (candidate / "data/profile.json").is_file():
            return candidate
    raise RuntimeError("Cannot locate repository; pass --root explicitly")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_json(path: Path):
    return json.loads(path.read_text("utf-8"))


def file_metadata(path: Path, root: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular source file: {path}")
    before = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while hashing: {path}")
    return {"path": path.relative_to(root).as_posix(), "bytes": after.st_size, "sha256": sha.hexdigest()}


def excluded_source(path: Path, allow_analysis_png: bool = False) -> bool:
    return ("__pycache__" in path.parts or path.name == ".DS_Store"
            or (path.suffix.lower() in FORBIDDEN_SUFFIXES
                and not (allow_analysis_png and path.suffix.lower() == ".png")))


class Plan:
    """Prepare all bytes and provenance before making any destination changes."""

    def __init__(self, root: Path, output: Path):
        self.root = root
        self.output = output
        self.entries: dict[str, dict] = {}
        self.data: dict[str, bytes] = {}

    def add(self, destination: Path, data: bytes, mode: str, sources: list[dict], description: str = "",
            allow_analysis_png: bool = False):
        if destination.is_absolute() or ".." in destination.parts or destination == Path("README.md"):
            raise ValueError(f"Invalid managed destination: {destination}")
        if excluded_source(destination, allow_analysis_png):
            raise ValueError(f"Excluded content cannot be archived: {destination}")
        key = destination.as_posix()
        if key in self.entries:
            raise ValueError(f"Duplicate destination: {key}")
        self.data[key] = data
        self.entries[key] = {
            "destination": key,
            "destination_repository_path": (self.output / destination).relative_to(self.root).as_posix(),
            "mode": mode, "bytes": len(data), "sha256": digest(data), "sources": sources,
        }
        if description:
            self.entries[key]["derivation"] = description
        if allow_analysis_png and destination.suffix.lower() == ".png":
            self.entries[key]["content_kind"] = "generated_analysis_plot; not a paper/source image"

    def copy(self, source: Path, destination: Path, allow_analysis_png: bool = False):
        if source.is_symlink() or not source.is_file() or excluded_source(source, allow_analysis_png):
            raise ValueError(f"Expected allowed regular source file: {source}")
        before = source.stat()
        data = source.read_bytes()
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed while reading: {source}")
        meta = {"path": source.relative_to(self.root).as_posix(), "bytes": len(data), "sha256": digest(data)}
        self.add(destination, data, "copied", [meta], allow_analysis_png=allow_analysis_png)

    def derived(self, value, destination: Path, sources: list[dict], description: str):
        self.add(destination, json_bytes(value), "derived", sources, description)


def without_hourly(value):
    """Preserve aggregate and case data, removing only named hourly payloads."""
    if isinstance(value, dict):
        return {key: without_hourly(item) for key, item in value.items()
                if key not in ("hourly", "reference_hourly")}
    if isinstance(value, list):
        return [without_hourly(item) for item in value]
    return value


def archive_run(plan: Plan, run: Path) -> dict:
    config = read_json(run / "config.json")
    complete = read_json(run / "complete.json")
    episode = complete.get("episodes")
    if type(episode) is not int or episode <= 0 or episode != config.get("episodes"):
        raise ValueError(f"Completion/config episode mismatch: {run}")
    target = Path("runs") / run.name
    for name in ("config.json", "manifest.json", "complete.json", "training.csv"):
        plan.copy(run / name, target / name)
    for path in sorted(run.glob("effective*.json")):
        plan.copy(path, target / path.name)

    source_root = run / "source"
    manifest = read_json(run / "manifest.json")
    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"Missing frozen-source hashes: {run}")
    for name, expected in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe frozen-source manifest path: {name}")
        actual = file_metadata(source_root / relative, plan.root)["sha256"]
        if actual != expected:
            raise ValueError(f"Frozen source hash mismatch: {source_root / relative}")
    source_count = 0
    excluded = []
    for path in sorted(source_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Do not follow frozen-source symlinks: {path}")
        if not path.is_file():
            continue
        if excluded_source(path):
            excluded.append(path.relative_to(run).as_posix())
            continue
        plan.copy(path, target / "source" / path.relative_to(source_root))
        source_count += 1

    final_evaluation = run / f"evaluation_{episode:05d}.json"
    if not final_evaluation.is_file():
        raise FileNotFoundError(f"Completed run has no final development evaluation: {final_evaluation}")
    plan.copy(final_evaluation, target / final_evaluation.name)
    stages = []
    for path in sorted(run.glob("evaluation_*.json")):
        match = EVALUATION_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unrecognized development evaluation name: {path}")
        stage_episode = int(match.group(1))
        if stage_episode > episode:
            raise ValueError(f"Evaluation exceeds completed episode: {path}")
        if path == final_evaluation:
            continue
        original = path.read_bytes()
        value = json.loads(original)
        if not isinstance(value, dict):
            raise ValueError(f"Expected evaluation object: {path}")
        meta = {"path": path.relative_to(plan.root).as_posix(), "bytes": len(original), "sha256": digest(original)}
        summary = without_hourly(value)
        summary["_archive_derivation"] = {
            "source": meta, "episode": stage_episode,
            "removed_keys": ["hourly", "reference_hourly"],
            "note": "Derived development summary; aggregate metrics and non-hourly per-case fields retained. Not the original full evaluation file.",
        }
        dest = target / "development_summaries" / path.name
        plan.derived(summary, dest, [meta], "Remove hourly/reference_hourly fields recursively; retain all other data and add derivation metadata.")
        stages.append(dest.as_posix())

    holdouts = []
    for name in ("holdout_108.json", "holdout_500_108.json"):
        path = run / name
        if path.exists():
            plan.copy(path, target / name)
            holdouts.append(name)
    checkpoints = [file_metadata(path, plan.root) for path in sorted(run.glob("*.pt"))]
    final_checkpoint = (run / f"checkpoint_{episode:05d}.pt").relative_to(plan.root).as_posix()
    checkpoint_record = {
        "checkpoint_files_distributed": False,
        "note": "Paths refer to local source files; weights are not included in this Git archive. These hashes do not substitute for model weights.",
        "final_checkpoint_path": final_checkpoint,
        "final_checkpoint_present": any(item["path"] == final_checkpoint for item in checkpoints),
        "checkpoints": checkpoints,
    }
    plan.derived(checkpoint_record, target / "checkpoints.json", checkpoints,
                 "Record local .pt paths, sizes and SHA256 only; never copy checkpoint bytes.")
    return {
        "name": run.name, "source": run.relative_to(plan.root).as_posix(),
        "destination": target.as_posix(), "episodes": episode,
        "frozen_manifest_hashes_verified": len(hashes), "frozen_files_copied": source_count,
        "excluded_frozen_files": excluded,
        "final_development_full_file": (target / final_evaluation.name).as_posix(),
        "development_derived_files": stages, "full_holdouts": holdouts,
        "checkpoints_recorded": len(checkpoints),
    }


def archive_literature(plan: Plan, evidence: Path) -> list[str]:
    missing = []
    copied = {}
    for name in LITERATURE_NAMES:
        path = evidence / name
        if path.is_file():
            destination = Path("literature") / name
            plan.copy(path, destination)
            copied[name] = plan.entries[destination.as_posix()]["sha256"]
        else:
            missing.append(name)
    original = evidence / "evidence_manifest.json"
    if original.is_file():
        original_bytes = original.read_bytes()
        value = json.loads(original_bytes)
        recorded = value.get("local_evidence_sha256", {})
        if not isinstance(recorded, dict):
            raise ValueError("Literature local_evidence_sha256 must be an object")
        public = {key: item for key, item in value.items() if key != "local_evidence_sha256"}
        public["archive_notice"] = (
            "Derived public provenance manifest. Papers, XML, PDF, excerpts, images and supplement archives are not distributed. "
            "Local-only hashes record the original audit material and do not claim that those files are available in this repository. "
            "Use publisher/DOI URLs in parameter_evidence.json and renewable_share_supplement.json for public source attribution."
        )
        public["copied_metadata_sha256"] = copied
        public["original_local_hash_records"] = [
            {"original_local_path": name, "recorded_sha256": sha,
             "included_in_archive": name in copied,
             "archive_path": f"literature/{name}" if name in copied else None,
             "recorded_hash_matches_copied_file": copied[name] == sha if name in copied else None}
            for name, sha in sorted(recorded.items())
        ]
        public["source_manifest"] = {
            "path": original.relative_to(plan.root).as_posix(),
            "bytes": len(original_bytes), "sha256": digest(original_bytes),
        }
        plan.derived(public, Path("literature/evidence_manifest.json"), [public["source_manifest"]],
                     "Preserve original audit metadata/hash records; explicitly distinguish distributed metadata from excluded local source material and record current copied-file hashes.")
    else:
        missing.append("evidence_manifest.json")
    return missing


def build_plan(root: Path, output: Path):
    plan = Plan(root, output)
    completed, skipped = [], []
    for run in sorted((root / "runs").glob("sac_revision_*")):
        if not run.is_dir():
            continue
        if run.is_symlink():
            raise ValueError(f"Do not follow run symlinks: {run}")
        if not (run / "complete.json").is_file():
            skipped.append(run.relative_to(root).as_posix())
            continue
        completed.append(archive_run(plan, run))
    if not completed:
        raise ValueError("No completed runs/sac_revision_* studies found")
    audit = root / "checks/sac_revision"
    for path in sorted(audit.iterdir()):
        if not path.is_file():
            continue
        if path.name == "final_selection.json":
            plan.copy(path, Path(path.name))
        elif path.name in FINAL_ANALYSIS_JSON_NAMES or path.name == "final_tests.log":
            plan.copy(path, Path("analysis") / path.name)
        elif path.name in DIAGNOSTIC_NAMES:
            plan.copy(path, Path("diagnostics") / path.name)
    for name in FINAL_ANALYSIS_DIRECTORIES:
        directory = audit / name
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in (".json", ".png"):
                    plan.copy(path, Path("analysis") / name / path.relative_to(directory),
                              allow_analysis_png=True)
    summary = audit / "study_summary"
    if summary.is_dir():
        for path in sorted(summary.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".csv"):
                plan.copy(path, Path("study_summary") / path.relative_to(summary))
    missing_literature = archive_literature(plan, audit / "literature")
    files = [plan.entries[key] for key in sorted(plan.entries)]
    manifest = {
        "schema_version": 1, "study": STUDY_NAME,
        "destination": output.relative_to(root).as_posix(),
        "policy": {
            "run_selection": "Only runs/sac_revision_* with complete.json and matching configured/completed episodes.",
            "evaluation": "Final development and named holdouts copied in full; earlier development files are explicitly derived summaries without hourly payloads.",
            "frozen_source": "Byte-for-byte; manifest hashes checked first. Caches, checkpoints and source-document/media formats excluded.",
            "literature": "Metadata/provenance only. No PDF, XML, raw excerpts, images or supplement archives.",
            "analysis": "Generated JSON/PNG from the two named final analysis directories and final evaluation/test records are copied byte-for-byte. The image exception does not apply to literature source images.",
            "diagnostics": "Explicit response-probe script/JSON allowlist only; intermediate trial *_analysis.json files are excluded because they can be regenerated from retained final development evaluations.",
            "probe_execution": "Probe scripts preserved byte-for-byte. Restore to original checks/sac_revision paths before running scripts whose ROOT depends on parents[2].",
            "README": "Not generated or modified by this script.",
            "manifest_self_hash": "This manifest cannot include its own content hash; all copied and other derived managed files are listed below.",
        },
        "completed_runs": completed, "skipped_incomplete_runs": skipped,
        "optional_literature_not_present": missing_literature,
        "copied_files": sum(item["mode"] == "copied" for item in files),
        "derived_files": sum(item["mode"] == "derived" for item in files),
        "managed_bytes_excluding_this_manifest": sum(item["bytes"] for item in files),
        "files": files,
    }
    return plan, manifest


def ensure_safe_destination(output: Path, relative: str):
    path = output / relative
    current = path
    while current != output.parent:
        if current.is_symlink():
            raise ValueError(f"Do not write through destination symlink: {current}")
        current = current.parent


def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".archive-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def apply_plan(plan: Plan, manifest: dict, dry_run: bool):
    manifest_path = plan.output / "archive_manifest.json"
    ensure_safe_destination(plan.output, "archive_manifest.json")
    old = read_json(manifest_path) if manifest_path.exists() else {}
    old_files = {item["destination"] for item in old.get("files", [])}
    stale = old_files - set(plan.entries)
    if stale:
        raise RuntimeError("Previously managed files would become stale; no deletion is automatic: " + ", ".join(sorted(stale)))
    changed = []
    for relative, data in plan.data.items():
        ensure_safe_destination(plan.output, relative)
        destination = plan.output / relative
        if destination.exists() and destination.read_bytes() == data:
            continue
        if destination.exists() and relative not in old_files:
            raise FileExistsError(f"Refusing to replace an unmanaged existing file: {destination}")
        changed.append(relative)
    if not dry_run:
        for relative in changed:
            atomic_write(plan.output / relative, plan.data[relative])
        encoded = json_bytes(manifest)
        if not manifest_path.exists() or manifest_path.read_bytes() != encoded:
            atomic_write(manifest_path, encoded)
    return {
        "dry_run": dry_run, "destination": str(plan.output),
        "completed_runs": len(manifest["completed_runs"]),
        "skipped_incomplete_runs": manifest["skipped_incomplete_runs"],
        "copied_files": manifest["copied_files"], "derived_files": manifest["derived_files"],
        "changed_files_excluding_manifest": len(changed),
        "managed_bytes_excluding_manifest": manifest["managed_bytes_excluding_this_manifest"],
        "optional_literature_not_present": manifest["optional_literature_not_present"],
        "README_modified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, help="Destination below this repository's experiments directory")
    parser.add_argument("--dry-run", action="store_true", help="Hash and validate inputs, report the plan, write nothing")
    args = parser.parse_args()
    root = (args.root or default_root()).resolve()
    requested = args.output or Path("experiments") / STUDY_NAME
    output = requested if requested.is_absolute() else root / requested
    if output.is_symlink():
        raise ValueError("Archive destination must not be a symlink")
    output = output.resolve()
    experiments = (root / "experiments").resolve()
    if output == experiments or not output.is_relative_to(experiments):
        raise ValueError("Destination must be a dedicated directory below repository experiments/")
    plan, manifest = build_plan(root, output)
    print(json.dumps(apply_plan(plan, manifest, args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
