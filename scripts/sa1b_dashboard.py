"""Thread-safe terminal status for SA-1B preparation (Rich is optional without a TTY)."""

from collections import deque
from threading import RLock
import time


class Dashboard:
    def __init__(self, destination, enabled=False, console=None):
        self.destination = destination
        self.enabled = enabled
        self.console = console
        self.lock = RLock()
        self.live = None
        self.source = "Waiting for source"
        self.source_bytes = 0
        self.source_total = None
        self.source_started = time.monotonic()
        self.shard = "Waiting for samples"
        self.samples = 0
        self.capacity = 1000
        self.timing_count = 0
        self.image_seconds_total = 0.0
        self.mask_seconds_total = 0.0
        self.annotations_total = 0
        self.status = "Starting"
        self.uploads = {}
        self.uploaded = 0
        self.recent = deque(maxlen=5)

    def __enter__(self):
        if self.enabled:
            try:
                from rich.console import Console
                from rich.live import Live
            except ImportError:
                raise RuntimeError("Install the dashboard with: uv sync --extra data-prep; or use --no-dashboard") from None
            self.live = Live(console=self.console or Console(stderr=True),
                             get_renderable=self.render, refresh_per_second=4)
            self.live.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.set_status("Interrupted" if exc_type is KeyboardInterrupt else "Failed" if exc_type else "Done")
        if self.live:
            self.live.stop()

    def set_status(self, status):
        with self.lock:
            self.status = status

    def download(self, name, completed, total, started):
        with self.lock:
            self.source, self.source_bytes, self.source_total = name, completed, total
            self.source_started = started

    def fill(self, name, completed, total):
        with self.lock:
            self.shard, self.samples, self.capacity = name, completed, total
            self.status = "Writing shards"

    def timing(self, timing):
        with self.lock:
            self.timing_count += 1
            self.image_seconds_total += timing["image_seconds"]
            self.mask_seconds_total += timing["mask_seconds"]
            self.annotations_total += timing["num_annotations"]

    def queued(self, name, size):
        with self.lock:
            self.uploads[name] = {"status": "Queued", "bytes": 0, "size": size}

    def upload_status(self, name, status, reset=False):
        with self.lock:
            self.uploads[name]["status"] = status
            if reset:
                self.uploads[name]["bytes"] = 0

    def advance(self, name, amount):
        with self.lock:
            self.uploads[name]["bytes"] += amount

    def completed(self, name):
        with self.lock:
            self.uploads.pop(name, None)
            self.uploaded += 1
            self.recent.append(name)

    def render(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.progress_bar import ProgressBar
        from rich.table import Table
        from rich.text import Text

        with self.lock:
            queued = sum(item["status"] == "Queued" for item in self.uploads.values())
            active = sum(item["status"] not in ("Queued", "Failed") for item in self.uploads.values())
            summary = Table.grid(padding=(0, 2))
            summary.add_row("Destination", Text(self.destination))
            summary.add_row("Status", self.status)
            summary.add_row("Uploads", f"{self.uploaded} uploaded | {active} active | {queued} queued")
            elapsed = max(time.monotonic() - self.source_started, 0.001)
            source_size = f"{self.source_bytes / 1e6:.1f} MB"
            if self.source_total:
                source_size += f" / {self.source_total / 1e6:.1f} MB ({self.source_bytes / self.source_total:.1%})"
            summary.add_row("Source", Text(self.source))
            summary.add_row("Downloaded", f"{source_size} | {self.source_bytes / elapsed / 1e6:.1f} MB/s")
            summary.add_row("Current shard", Text(self.shard))
            summary.add_row("Filled", f"{self.samples:,}/{self.capacity:,} images ({self.samples / self.capacity:.1%})")
            summary.add_row("", ProgressBar(total=self.capacity, completed=self.samples))
            if self.timing_count:
                avg_image_ms = self.image_seconds_total / self.timing_count * 1000
                avg_mask_ms = self.mask_seconds_total / self.timing_count * 1000
                avg_annotations = self.annotations_total / self.timing_count
                summary.add_row(
                    "Encode time",
                    f"{avg_image_ms:.1f} ms image | {avg_mask_ms:.1f} ms masks/image "
                    f"({avg_annotations:.1f} masks/image avg)",
                )
            uploads = Table("Shard", "State", "Progress", expand=True)
            for name, item in self.uploads.items():
                progress = min(1, max(0, item["bytes"] / max(1, item["size"])))
                uploads.add_row(Text(name), item["status"], f"{progress:.1%}")
            for name in reversed(self.recent):
                uploads.add_row(Text(name), "Uploaded", "100.0%")
            return Panel(Group(summary, uploads), title="SA-1B WebDataset", border_style="cyan")
