"""Resolve automatic resume within one experiment's local checkpoint files."""

from pathlib import Path


def resolve_resume_checkpoint(resume_from, final_checkpoint):
    """Use save time across periodic, epoch, and final checkpoints for this run."""
    if resume_from not in ("auto", "latest"):
        return resume_from

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
        print(f"No local checkpoint found for {final.stem}; starting fresh")
        return None
    latest = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    latest = str(latest.resolve())
    print(f"Automatically resuming from {latest}")
    return latest
