from pathlib import Path
import numpy as np
import sys
import os
from pathlib import Path

# 获取当前脚本的绝对路径
current_dir = Path(__file__).resolve().parent
# 获取项目根目录 (即 scripts 的上一级)
root_dir = current_dir.parent
# 将根目录添加到 Python 搜索路径中
sys.path.append(str(root_dir))
from src.envs.liionpack_spme_pack_env import build_pack_env
from src.utils.config import load_config

cfg = load_config("configs/pack_3p6s_spme_with_soh_prior.yaml").data
env = build_pack_env(cfg["env"])

obs, info = env.reset()
print("Reset: Vpack?", info.get("V_pack"), "Vcell_max", info["V_cell_max"])

# lo = float(env.action_space.low[0])
# hi = float(env.action_space.high[0])
# I0 = min(abs(lo), abs(hi), 10.0)  # 取 10A 或者更小，避免直接触发电压限
I0 = 10.0

for I in [ +I0, -I0 ]:
    obs, r, terminated, truncated, info = env.step(np.array([I], dtype=np.float32))

    print(f"I={I:+.2f}A -> Vcell_max={info['V_cell_max']:.4f}, Vcell_min={info['V_cell_min']:.4f}, V_pack={info.get('V_pack')}")
