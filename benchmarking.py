"""Benchmark the broad phase (hash grid) and narrow phase (model), without rendering."""

import argparse
from time import perf_counter

import numpy as np
import torch

import mesh_intersection as collision


def clock_ms(detector):
    if detector.is_cuda:
        torch.cuda.synchronize(detector.device)
    return perf_counter() * 1000


def time_phases(sim):
    start = clock_ms(sim.detector)
    pairs = sim.broad_phase()
    split = clock_ms(sim.detector)
    hits = sim.narrow_phase(pairs)
    end = clock_ms(sim.detector)
    return len(pairs), len(hits), (split - start, end - split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", nargs="?", help="Input OBJ (default: simulator mesh)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--at-steps", type=int, nargs="+", help="Pose indices (default: all)")
    parser.add_argument("--repeat", type=int, default=1, help="Timed runs per pose; report medians")
    args = parser.parse_args()
    runtime = collision.RuntimeConfig.from_args(collision.parse_args(["--cpu"] if args.cpu else []))
    config = runtime.simulation
    steps = args.at_steps or range(config.num_steps + 1)
    if args.repeat < 1 or any(not 0 <= step <= config.num_steps for step in steps):
        parser.error(f"--repeat must be positive and steps must be in [0, {config.num_steps}]")

    detector = collision.CollisionDetector(runtime.model_path, runtime.transform_names, runtime.device, runtime.dtype)
    static = collision.TetrahedralMesh.from_file(args.mesh or runtime.static_mesh_path, "A", [0, 0, 1])
    moving = collision.TetrahedralMesh("B", [0, 1, 0], static.vertices + config.initial_offset, static.tetrahedra)
    sim = collision.TetrahedralCollisionSimulator(config, detector, static, moving)
    print(f"device={runtime.device} dtype={runtime.dtype} threads={torch.get_num_threads()} "
          f"batch={config.batch_size} tets={len(static.tetrahedra):,} repeat={args.repeat}")

    totals = np.zeros(2)
    print("step   candidates       hits     broad_ms    narrow_ms")
    for step in steps:
        offset = config.initial_offset * (1 - step / config.num_steps)
        moving.set_vertices(static.vertices + offset)
        # Warm both phases with real candidates, at most one model batch.
        sim.narrow_phase(sim.broad_phase()[:config.batch_size])
        samples = []
        for _ in range(args.repeat):
            candidates, hits, durations = time_phases(sim)
            samples.append(durations)
        broad_ms, narrow_ms = np.median(samples, axis=0)
        totals += (broad_ms, narrow_ms)
        print(f"{step:4} {candidates:12,} {hits:10,} {broad_ms:12.2f} {narrow_ms:12.2f}", flush=True)
    print(f"Total across {len(steps)} poses: broad={totals[0]:.2f} ms  narrow={totals[1]:.2f} ms")


if __name__ == "__main__":
    main()
