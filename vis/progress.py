"""Progress display for fresh and mid-epoch resumed training."""

from pytorch_lightning.callbacks import TQDMProgressBar


class ResumeProgressBar(TQDMProgressBar):
    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self._needs_setup = True

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self._needs_setup = True

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self._needs_setup:
            return
        # Lightning skips the epoch-start hook when resuming mid-epoch.
        super().on_train_epoch_start(trainer, pl_module)
        bar = self.train_progress_bar
        # tqdm subtracts initial when computing speed and remaining time.
        bar.initial = batch_idx
        bar.n = batch_idx
        bar.last_print_n = batch_idx
        bar.refresh()
        self._needs_setup = False
