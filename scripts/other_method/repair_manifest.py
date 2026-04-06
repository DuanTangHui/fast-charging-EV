import json
from pathlib import Path

# 设置路径（请根据你的实际运行目录调整）
models_dir = Path("runs/other_method/models")
manifest_path = models_dir / "manifest.json"

# 1. 如果文件已存在，先读取现有内容；否则创建空字典
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
else:
    manifest = {}

# 2. 扫描并添加 RL 算法 (td3, ppo, ddpg)
# 代码逻辑中 RL 的路径通常是：models/{algo}_static_gp/agent_ckpt.pt
for algo in ["td3", "ppo", "ddpg"]:
    algo_dir = models_dir / f"{algo}_static_gp"
    ckpt_path = algo_dir / "agent_ckpt.pt"
    
    if ckpt_path.exists():
        manifest[algo] = {
            "type": "rl_agent",
            "algorithm": algo,
            "path": str(ckpt_path)
        }
        print(f"✅ 已找到并添加 {algo} 到 manifest")
    else:
        print(f"⚠️ 未找到 {algo} 的模型文件，跳过")

# 3. 确保基础方法也在里面（如果文件丢了的话）
for base_method in ["multistage_cc_schedule.json", "ga_schedule.json"]:
    method_name = base_method.replace("_schedule.json", "")
    if method_name == "multistage_cc_schedule": method_name = "multistage_cc" # 修正命名习惯
    
    file_path = models_dir / base_method
    if file_path.exists():
        # 根据你的脚本逻辑，这里 key 可能是 multistage_cc 或 ga_cc
        key = "multistage_cc" if "multistage" in base_method else "ga_cc"
        manifest[key] = {
            "type": "schedule",
            "path": str(file_path)
        }

# 4. 写回文件
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n🚀 修复完成！文件已保存至: {manifest_path}")