# """Plot SOC-OCV-current limit surface using Chen2020 OCV model."""
# from __future__ import annotations

# import argparse
# from typing import Callable
# import numpy as np
# import matplotlib.pyplot as plt


# def get_chen2020_ocv_func() -> Callable[[np.ndarray], np.ndarray]:
#     """Return OCV(SOC) interpolation based on Chen2020 parameters."""
#     import pybamm

#     param = pybamm.ParameterValues("Chen2020")
#     u_p = param["Positive electrode OCP [V]"]
#     u_n = param["Negative electrode OCP [V]"]

#     try:
#         sto_n_0 = param["Lower stoichiometric limit in negative electrode"]
#         sto_n_1 = param["Upper stoichiometric limit in negative electrode"]
#         sto_p_0 = param["Upper stoichiometric limit in positive electrode"]
#         sto_p_1 = param["Lower stoichiometric limit in positive electrode"]
#     except KeyError:
#         sto_n_0, sto_n_1 = 0.0279, 0.9014
#         sto_p_0, sto_p_1 = 0.9077, 0.2661

#     soc_range = np.linspace(0.0, 1.0, 200)
#     ocv_values: list[float] = []
#     for soc in soc_range:
#         curr_sto_n = sto_n_0 + soc * (sto_n_1 - sto_n_0)
#         curr_sto_p = sto_p_0 - soc * (sto_p_0 - sto_p_1)
#         ocv = (
#             param.evaluate(u_p(pybamm.Scalar(curr_sto_p)))
#             - param.evaluate(u_n(pybamm.Scalar(curr_sto_n)))
#         )
#         ocv_values.append(float(ocv))

#     ocv_array = np.asarray(ocv_values, dtype=float)

#     def ocv_func(soc: np.ndarray | float) -> np.ndarray:
#         soc_arr = np.asarray(soc, dtype=float)
#         return np.interp(soc_arr, soc_range, ocv_array)

#     return ocv_func


# def build_soc_ocv_current_surface(
#     soc_min: float = 0.2,
#     soc_max: float = 1.0,
#     n_soc: int = 24,
#     n_current: int = 40,
#     v_limit: float = 4.2,
#     r_internal: float = 0.025,
#     n_parallel: float = 3.0,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Build SOC-OCV-I surface arrays for plotting I values up to I_max."""
#     soc_line = np.linspace(soc_min, soc_max, n_soc)
#     ocv_line = np.asarray(get_chen2020_ocv_func()(soc_line), dtype=float)

#     i_single = np.maximum(0.0, (v_limit - ocv_line) / r_internal)
#     i_pack = i_single * n_parallel

#     current_ratio = np.linspace(0.0, 1.0, n_current)
#     soc_mesh = np.tile(soc_line[None, :], (n_current, 1))
#     ocv_mesh = np.tile(ocv_line[None, :], (n_current, 1))
#     current_mesh = current_ratio[:, None] * i_pack[None, :]
#     return soc_mesh, ocv_mesh, current_mesh


# def plot_surface(save_path: str | None = None) -> None:
#     """Render the SOC-OCV-current 3D surface (filled below I_max)."""
#     soc_mesh, ocv_mesh, current_mesh = build_soc_ocv_current_surface()

#     fig = plt.figure(figsize=(9, 5.5))
#     ax = fig.add_subplot(111, projection="3d")
#     surf = ax.plot_surface(
#         soc_mesh,
#         ocv_mesh,
#         current_mesh,
#         cmap="viridis",
#         edgecolor="#5a5a5a",
#         linewidth=0.5,
#         antialiased=True,
#         alpha=0.95,
#     )

#     ax.set_xlabel("SOC", labelpad=8)
#     ax.set_ylabel("OCV (V)", labelpad=10)
#     ax.set_zlabel("电流 (A)", labelpad=8)
#     ax.set_title("SOC-OCV-充电电流约束曲面")
#     ax.view_init(elev=18, azim=-155)
#     ax.set_ylim(float(np.min(ocv_mesh)), float(np.max(ocv_mesh)))
#     ax.set_zlim(0.0, max(1.0, float(np.max(current_mesh)) * 1.05))
#     fig.colorbar(surf, shrink=0.65, pad=0.08, label="I_max (A)")

#     # 叠加最大电流边界线，便于区分“上边界”与“边界以下可行电流区域”
#     ax.plot(
#         soc_mesh[-1, :],
#         ocv_mesh[-1, :],
#         current_mesh[-1, :],
#         color="black",
#         linewidth=2.0,
#         label="I_max boundary",
#     )
#     ax.legend(loc="upper right")

#     plt.tight_layout()
#     if save_path:
#         fig.savefig(save_path, dpi=160)
#         print(f"[OK] saved figure to {save_path}")
#     else:
#         plt.show()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Plot SOC-OCV-current surface")
#     parser.add_argument("--save-path", type=str, default=None, help="Optional output image path")
#     args = parser.parse_args()
#     plot_surface(save_path=args.save_path)

"""
Plot current-vs-SOC (and optional SOC-OCV-current surface) 
using simulated Chen2020 OCV data for a clean, specific layout.
"""
from __future__ import annotations

import argparse
import os
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

# --- 字体、单位与全局设置 ---
# 设置中文字体（示例：Droid Sans Fallback），确保兼容性
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Droid Sans Fallback', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 字体大小设置：统一设置为 8pt
plt.rcParams['font.size'] = 8 

# 厘米转英寸函数
def cm_to_inches(cm: float) -> float:
    return cm / 2.54

# 2. 图片大小设置：设定为 3cm (宽) x 2.5cm (高) (W, H)
FIG_SIZE = (cm_to_inches(7), cm_to_inches(5)) 


def get_mock_chen2020_ocv_func() -> Callable[[np.ndarray], np.ndarray]:
    """Return simulated OCV(SOC) interpolation based on Chen2020 (LGM50) approximation."""
    soc_range = np.linspace(0.0, 1.0, 200)
    # LGM50 近似：3.4V (0% SOC) 到 4.2V (100% SOC)
    ocv_array = 3.42 + 0.6 * soc_range + 0.15 * (soc_range**2) + 0.03 * np.sin(2 * np.pi * soc_range)

    def ocv_func(soc: np.ndarray | float) -> np.ndarray:
        soc_arr = np.asarray(soc, dtype=float)
        return np.interp(soc_arr, soc_range, ocv_array)

    return ocv_func


def build_current_vs_soc(
    soc_min: float = 0.2,
    soc_max: float = 1.0,
    n_soc: int = 120,
    v_limit: float = 4.2,
    r_internal: float = 0.025,
    n_parallel: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build SOC-OCV-Imax line arrays for plotting current change vs SOC."""
    soc_line = np.linspace(soc_min, soc_max, n_soc)
    ocv_line = np.asarray(get_mock_chen2020_ocv_func()(soc_line), dtype=float)
    i_single = np.maximum(0.0, (v_limit - ocv_line) / r_internal)
    i_pack = i_single * n_parallel
    return soc_line, ocv_line, i_pack


def build_soc_ocv_current_surface(
    soc_min: float = 0.2,
    soc_max: float = 1.0,
    n_soc: int = 24,
    n_current: int = 40,
    v_limit: float = 4.2,
    r_internal: float = 0.025,
    n_parallel: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build SOC-OCV-I surface arrays for plotting I values up to I_max."""
    soc_line = np.linspace(soc_min, soc_max, n_soc)
    ocv_line = np.asarray(get_mock_chen2020_ocv_func()(soc_line), dtype=float)

    i_single = np.maximum(0.0, (v_limit - ocv_line) / r_internal)
    i_pack = i_single * n_parallel

    current_ratio = np.linspace(0.0, 1.0, n_current)
    soc_mesh = np.tile(soc_line[None, :], (n_current, 1))
    ocv_mesh = np.tile(ocv_line[None, :], (n_current, 1))
    current_mesh = current_ratio[:, None] * i_pack[None, :]
    return soc_mesh, ocv_mesh, current_mesh


def plot_current_vs_soc_cleaned(save_path: str | None = None) -> None:
    """Render 2D figure: current limit vs SOC, cleaned and small."""
    soc_line, _, i_pack = build_current_vs_soc()

    fig, ax = plt.subplots(figsize=FIG_SIZE) # 使用设定的 3x2.5cm
    
    ax.plot(soc_line, i_pack, color="#1f77b4", linewidth=1.2, label="I_max")
    ax.fill_between(soc_line, 0.0, i_pack, color="#1f77b4", alpha=0.15)
    
    ax.set_xlabel("SOC") # 字体大小已全局设定为 8pt
    ax.set_ylabel("电流 (A)")
    ax.set_title("电流随 SOC 的变化")
    
    # 3. 去除背景线条：去除网格线
    ax.grid(False) 

    ax.set_xlim(float(np.min(soc_line)), float(np.max(soc_line)))
    ax.set_ylim(0.0, max(1.0, float(np.max(i_pack)) * 1.05))
    
    # 缩减图例以适应小图像
    # ax.legend(loc="upper right", fontsize=7) 

    plt.tight_layout(pad=0.2) # 调整 padding

    if save_path:
        # 强制保存为 svg
        if not save_path.lower().endswith('.svg'):
            save_path = os.path.splitext(save_path)[0] + '.svg'
        fig.savefig(save_path, format='svg', bbox_inches='tight') # 强制使用指定格式和 bbox
        print(f"[OK] 2D 图像已保存至 {save_path}")
    else:
        plt.show()


def plot_surface_cleaned(save_path: str | None = None) -> None:
    """Render the SOC-OCV-current 3D surface, cleaned and small."""
    soc_mesh, ocv_mesh, current_mesh = build_soc_ocv_current_surface()

    fig = plt.figure(figsize=FIG_SIZE) # 使用设定的 3x2.5cm
    ax = fig.add_subplot(111, projection="3d")
    
    # 3. 去除背景线条：去除 3D 绘图底板背景和网格线
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    surf = ax.plot_surface(
        soc_mesh,
        ocv_mesh,
        current_mesh,
        cmap="viridis",
        edgecolor="#5a5a5a",
        linewidth=0.3, # 缩细边缘线
        antialiased=True,
        alpha=0.9,
    )

    # 字体大小设置：统一 8pt
    ax.set_xlabel("SOC", labelpad=2)
    ax.set_ylabel("OCV (V)", labelpad=3)
    ax.set_zlabel("电流 (A)", labelpad=2)
    
    # 3D 视角的微调
    ax.view_init(elev=20, azim=-140)
    
    ax.set_ylim(float(np.min(ocv_mesh)), float(np.max(ocv_mesh)))
    ax.set_zlim(0.0, max(1.0, float(np.max(current_mesh)) * 1.05))

    # 缩减 Colorbar 以适应小图像
    # cbar = fig.colorbar(surf, shrink=0.4, pad=0.01)
    # cbar.ax.tick_params(labelsize=7)

    # 叠加最大电流边界线
    ax.plot(
        soc_mesh[-1, :],
        ocv_mesh[-1, :],
        current_mesh[-1, :],
        color="black",
        linewidth=1.2,
        label="I_max",
    )
    # ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout(pad=0.1)
    if save_path:
        if not save_path.lower().endswith('.svg'):
            save_path = os.path.splitext(save_path)[0] + '.svg'
        fig.savefig(save_path, format='svg', bbox_inches='tight') # 强制格式和 bbox
        print(f"[OK] 3D 图像已保存至 {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot cleaned current-vs-SOC or SOC-OCV-current surface")
    parser.add_argument(
        "--mode",
        type=str,
        default="current-vs-soc",
        choices=["current-vs-soc", "surface"],
        help="Plot mode. Default: current-vs-soc",
    )
    # 设置默认文件名以满足 SVG 要求
    parser.add_argument("--save-path", type=str, default="output.svg", help="Optional output image path")
    args = parser.parse_args()
    
    if args.mode == "surface":
        plot_surface_cleaned(save_path=args.save_path)
    else:
        plot_current_vs_soc_cleaned(save_path=args.save_path)