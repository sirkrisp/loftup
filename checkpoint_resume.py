"""Resolve automatic resume within one experiment's local or HF checkpoint files."""

from pathlib import Path


def resolve_resume_checkpoint(resume_from, final_checkpoint, source="local", repo_id=None, prefix=None):
    """Use save time across periodic, epoch, and final checkpoints for this run."""
    if resume_from not in ("auto", "latest"):
        return resume_from
    if source == "hf":
        return latest_hf_checkpoint(repo_id, prefix)
    if source != "local":
        raise ValueError("resume_source must be local or hf")

    final = Path(final_checkpoint).expanduser()
    candidates = [final] if final.is_file() else []
    epoch_dir = final.with_suffix("")
    if epoch_dir.is_dir():
        candidates.extend(path for path in epoch_dir.glob("*.ckpt") if path.is_file())
    # Periodic files are siblings named <experiment>_<optimizer step>.ckpt.
    # Avoid a glob containing the experiment name: names may contain brackets.
    prefix = final.stem + "_"
    if final.parent.is_dir():
        candidates.extend(
            path for path in final.parent.iterdir()
            if path.is_file() and path.suffix == ".ckpt"
            and path.stem.startswith(prefix) and path.stem[len(prefix):].isdigit()
        )
    if not candidates:
        print(f"No local checkpoint found for {final.stem}")
        return None
    latest = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    latest = str(latest.resolve())
    print(f"Automatically resuming from {latest}")
    return latest


def resolve_stage1_checkpoint(pretrained, output_root, run_name, repo_id, source="auto"):
    """Find Stage 1 weights locally, or download that experiment from HF."""
    if pretrained != "auto":
        return pretrained
    if source not in {"auto", "local", "hf"}:
        raise ValueError("stage1_source must be auto, local, or hf")
    final = Path(output_root) / "checkpoints" / "loftup_stage1" / f"{run_name}.ckpt"
    if source != "hf":
        checkpoint = resolve_resume_checkpoint("auto", final)
        if checkpoint is not None:
            print(f"Initializing Stage 2 from Stage 1 checkpoint: {checkpoint}")
            return checkpoint
        if source == "local":
            raise FileNotFoundError(f"No Stage 1 checkpoint found for {run_name}")
    return latest_hf_checkpoint(repo_id, f"stage1/{run_name}")


def latest_hf_checkpoint(repo_id, prefix):
    """Download the newest uploaded checkpoint for one experiment, pinned to a commit."""
    from huggingface_hub import HfApi, hf_hub_download

    if not repo_id or not prefix:
        raise ValueError("HF resume requires a repository and experiment prefix")
    api = HfApi()
    revision = api.repo_info(repo_id, repo_type="model").sha
    candidates = [entry for entry in api.list_repo_tree(
        repo_id, repo_type="model", path_in_repo=prefix, revision=revision,
        recursive=True, expand=True,
    ) if entry.path.startswith(prefix.rstrip('/') + '/') and entry.path.endswith('.ckpt')]
    if not candidates:
        raise FileNotFoundError(f"No HF checkpoints found in {repo_id}/{prefix}")
    latest = max(candidates, key=lambda entry: (entry.last_commit.date, entry.path))
    print(f"Downloading resume checkpoint from HF: {repo_id}/{latest.path}")
    return hf_hub_download(repo_id, latest.path, repo_type="model", revision=revision)
