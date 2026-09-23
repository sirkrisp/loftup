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
    parser.add_argument("--ddp", action="store_true",
                        help="Also test a 32 MiB collective and DDP startup/backward")
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
               f"NCCL build version={torch.cuda.nccl.version()} "
               "(NCCL_DEBUG=INFO reports the loaded runtime version)")

    report("Entering init_process_group")
    dist.init_process_group(args.backend, timeout=timedelta(seconds=60))
    report("Process group created; entering all_reduce")
    value = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(value)
    actual = value.item()
    expected = world * (world + 1) / 2
    if actual != expected:
        raise RuntimeError(f"all_reduce returned {actual}, expected {expected}")
    report(f"all_reduce passed: {actual}")
    if args.ddp:
        report("Entering 32 MiB all_reduce")
        large = torch.full((8 * 1024 * 1024,), float(rank + 1), device=device)
        dist.all_reduce(large)
        if not torch.all(large == expected).item():
            raise RuntimeError("Large all_reduce returned incorrect values")
        del large
        report("32 MiB all_reduce passed; creating model")
        torch.manual_seed(rank)
        model = torch.nn.Sequential(
            torch.nn.Linear(2048, 2048), torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
        ).to(device)
        # Exercise a frozen backbone plus trainable parameters, as in LoftUp.
        model[0].requires_grad_(False)
        report("Entering DDP constructor (parameter synchronization)")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if args.backend == "nccl" else None,
            find_unused_parameters=True,
        )
        report("DDP constructor passed; entering forward/backward")
        model(torch.randn(2, 2048, device=device)).square().mean().backward()
        for parameter in model.parameters():
            if parameter.requires_grad:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                    raise RuntimeError("Missing or nonfinite DDP gradient")
        if args.backend == "nccl":
            torch.cuda.synchronize()
        report("DDP forward/backward passed")
        del model
    report("Destroying process group")
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()
    report("PASS")


if __name__ == "__main__":
    main()
