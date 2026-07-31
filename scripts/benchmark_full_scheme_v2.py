from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from active_star_ris.full_scheme_v2.config import load_config
from active_star_ris.full_scheme_v2.environment import ActiveStarRisKeyEnvironment


def main() -> int:
    parser = argparse.ArgumentParser(description="测量真实配置下的环境步速度。")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/full_scheme_v2_paper_corrected.yaml",
    )
    parser.add_argument(
        "--architecture",
        default="partially_active_fixed",
        choices=[
            "passive",
            "partially_active_fixed",
            "partially_active_dynamic",
            "fully_active_fixed",
        ],
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objective-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config = replace(
        config,
        environment=replace(
            config.environment,
            architecture=args.architecture,
            seed=args.seed,
        ),
    )
    if args.objective_samples is not None:
        config = replace(
            config,
            robust=replace(
                config.robust,
                objective_samples=args.objective_samples,
            ),
        )

    env = ActiveStarRisKeyEnvironment(config)
    state, _ = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    print("首次调用会包含 Numba JIT 编译，先执行一个预热步。", flush=True)
    action = rng.uniform(-1.0, 1.0, env.action_dimension).astype(np.float32)
    state, _, done, _, _ = env.step(action)
    if done:
        state, _ = env.reset()

    start = time.perf_counter()
    for _ in range(args.steps):
        action = rng.uniform(-1.0, 1.0, env.action_dimension).astype(np.float32)
        state, _, done, _, _ = env.step(action)
        if done:
            state, _ = env.reset()
    elapsed = time.perf_counter() - start
    speed = args.steps / max(elapsed, 1.0e-12)

    print(f"architecture={args.architecture}")
    print(f"samples_per_step={config.probing.samples_per_step}")
    print(f"objective_samples={config.robust.objective_samples}")
    print(f"elapsed={elapsed:.3f} s")
    print(f"speed={speed:.4f} step/s")
    print(f"单个 5000 步任务估计={5000.0 / speed / 3600.0:.2f} h")
    print(f"8 个预实验任务串行估计={8.0 * 5000.0 / speed / 3600.0:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
