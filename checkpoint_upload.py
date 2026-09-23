"""Upload every completed Lightning checkpoint to a private HF model repo."""

from datetime import datetime, timezone
from pathlib import Path
import time
from uuid import uuid4

from huggingface_hub import HfApi
from pytorch_lightning import Callback
from pytorch_lightning.plugins.io import TorchCheckpointIO


class HubCheckpointIO(TorchCheckpointIO):
    def __init__(self, repo_id, prefix, retries=3):
        super().__init__()
        if retries < 1:
            raise ValueError("hf.retries must be positive")
        self.repo_id, self.prefix, self.retries = repo_id, prefix, retries
        self._api = None

    def prepare(self):
        if self._api is not None:
            return
        api = HfApi()  # HF_TOKEN or the user's cached `hf auth login` token.
        api.create_repo(repo_id=self.repo_id, repo_type="model", private=True, exist_ok=True)
        if not api.repo_info(self.repo_id, repo_type="model").private:
            raise RuntimeError(f"Refusing to upload checkpoints to public repository {self.repo_id}")
        api.auth_check(repo_id=self.repo_id, repo_type="model")
        self._api = api

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        # Lightning's strategy invokes checkpoint I/O only on global rank zero.
        # Complete the local atomic save before reading it for upload.
        super().save_checkpoint(checkpoint, path, storage_options)
        for attempt in range(self.retries):
            try:
                self.prepare()
                self._api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=f"{self.prefix}/{Path(path).name}",
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"Save {self.prefix.split('/')[0]} checkpoint {Path(path).name}",
                )
                return
            except Exception as exc:
                if attempt + 1 == self.retries:
                    raise RuntimeError(
                        f"Checkpoint saved locally at {path}, but upload to {self.repo_id} failed "
                        f"after {self.retries} attempts"
                    ) from exc
                time.sleep(2 ** attempt)


class HubUploadPreflight(Callback):
    def __init__(self, checkpoint_io):
        self.checkpoint_io = checkpoint_io

    def on_fit_start(self, trainer, pl_module):
        error = None
        if trainer.is_global_zero:
            try:
                self.checkpoint_io.prepare()
            except Exception as exc:
                error = f"HF checkpoint upload preflight failed ({type(exc).__name__}): {exc}"
        error = trainer.strategy.broadcast(error)
        if error:
            raise RuntimeError(error)


def configure_checkpoint_upload(cfg, callbacks, stage, run_name):
    if not cfg.hf.enabled:
        return []
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    checkpoint_io = HubCheckpointIO(
        cfg.hf.repo_id, f"{stage}/{run_name}/{run_id}", cfg.hf.retries,
    )
    callbacks.append(HubUploadPreflight(checkpoint_io))
    return [checkpoint_io]
