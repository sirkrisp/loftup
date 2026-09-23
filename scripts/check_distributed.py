"""Check rank startup and collectives without model downloads or data loading.

Run with: python -m torch.distributed.run --standalone --nproc_per_node=2
          scripts/check_distributed.py
Use --backend=gloo to test CPU communication without initializing CUDA.
"""

import argparse
from datetime import timedelta
import faulthandler
import os
import socket

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])

    def report(message):
        print(f"[rank {rank}, pid {os.getpid()}] {message}", flush=True)

    # A native CUDA hang may prevent the process-group timeout from firing.
    # Emit Python stacks too; use an external `timeout -k` for a hard deadline.
    faulthandler.dump_traceback_later(60, repeat=True)
    report(f"host={socket.gethostname()} torch={torch.__version__} "
           f"cuda={torch.version.cuda} backend={args.backend} world={world}")
    for name in ("NCCL_P2P_DISABLE", "NCCL_CUMEM_HOST_ENABLE", "NCCL_DEBUG"):
        report(f"{name}={os.environ.get(name, '<unset>')}")

    device = torch.device("cpu")
    if args.backend == "nccl":
        report(f"Initializing CUDA device {local_rank}")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        probe = torch.ones(16, device=device)
        probe = probe + 1
        torch.cuda.synchronize()
        assert probe.sum().item() == 32
        report(f"CUDA operation passed: {torch.cuda.get_device_name(local_rank)}; "
               f"NCCL={torch.cuda.nccl.version()}")

    report("Entering init_process_group")
    dist.init_process_group(args.backend, timeout=timedelta(seconds=60))
    report("Process group created; entering all_reduce")
    value = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(value)
    actual = value.item()
    expected = world * (world + 1) / 2
    if actual != expected:
        raise RuntimeError(f"all_reduce returned {actual}, expected {expected}")
    report(f"all_reduce passed: {actual}; destroying process group")
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()
    report("PASS")


if __name__ == "__main__":
    main()
