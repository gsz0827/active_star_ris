from __future__ import annotations

import compileall
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from active_star_ris.full_scheme_v2.config import load_config
from active_star_ris.full_scheme_v2.environment import ActiveStarRisKeyEnvironment


def main() -> int:
    if not compileall.compile_dir(ROOT / "src/active_star_ris/full_scheme_v2", quiet=1):
        raise SystemExit("Python 编译检查失败")
    config = load_config(ROOT / "configs/full_scheme_v2_paper.yaml")
    config = replace(
        config,
        robust=replace(config.robust, objective_samples=2),
        probing=replace(config.probing, samples_per_step=24),
        environment=replace(config.environment, episode_length=2),
    )
    env = ActiveStarRisKeyEnvironment(config)
    state, _ = env.reset(seed=0)
    action = env.rng.uniform(-1.0, 1.0, env.action_dimension)
    next_state, reward, terminated, truncated, info = env.step(action)
    assert state.shape == next_state.shape == (env.state_dimension,)
    assert not truncated
    assert "mean_secure_key_rate_bps" in info
    print(f"环境烟雾检查通过，reward={reward:.6f}, terminated={terminated}")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests/test_full_scheme_v2.py")],
        cwd=ROOT,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
