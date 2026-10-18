import json, sys, time
from pathlib import Path
import numpy as np
import torch

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W))
(W / "checks").mkdir(exist_ok=True)
from eqnsac.config import Config
from eqnsac.environment import Environment
from eqnsac.agent import Agent
from eqnsac.replay import Replay

torch.set_num_threads(1)
rows = []
for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    cfg = Config(device=device)
    env = Environment()
    agent = Agent(cfg, env)
    replay = Replay(1024, 5, 0.5, 919)
    state = env.reset(env.generator.sample(991))
    for n in range(500):
        a = np.array([0, 0], np.float32)
        ns, r, d, _ = env.step(a)
        replay.add(state, a, r, ns, d)
        state = ns
        if d:
            state = env.reset(env.generator.sample(992 + n))
    for _ in range(2):
        agent.update(replay)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        metrics = agent.update(replay)
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / 5
    rows.append(
        {
            "device": device,
            "seconds_per_update": seconds,
            "hidden": 256,
            "batch": 128,
            "quantiles": 32,
            "ensembles": "2x5",
            "measured_updates": 5,
            "finite": bool(torch.isfinite(metrics).all()),
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        }
    )
(W / "checks/短时性能测量.json").write_text(json.dumps(rows, indent=2), "utf-8")
print(json.dumps(rows, indent=2))
