import pybamm
import numpy as np
from scipy.interpolate import interp1d

def get_chen2020_ocv_func():
    # 1. 加载参数
    param = pybamm.ParameterValues("Chen2020")
    
    # 2. 从你提供的列表中锁定的准确键名
    U_p = param["Positive electrode OCP [V]"]
    U_n = param["Negative electrode OCP [V]"]
    
    # 获取锂离子计量比限制 (用于定义 SOC 0% 到 100%)
    # 如果下述四个键名报错，通常是因为 PyBaMM 版本中对极组容量限制的定义不同
    # 你可以先尝试以下标准 Chen2020 键名：
    try:
        sto_n_0 = param["Lower stoichiometric limit in negative electrode"]
        sto_n_1 = param["Upper stoichiometric limit in negative electrode"]
        sto_p_0 = param["Upper stoichiometric limit in positive electrode"]
        sto_p_1 = param["Lower stoichiometric limit in positive electrode"]
    except KeyError:
        # 如果报错，请直接手动指定 Chen2020 的典型计量比（LG M50 电池）
        sto_n_0, sto_n_1 = 0.0279, 0.9014
        sto_p_0, sto_p_1 = 0.9077, 0.2661
    
    # 3. 构建 SOC 映射
    soc_range = np.linspace(0, 1, 100)
    ocv_values = []
    
    for soc in soc_range:
        curr_sto_n = sto_n_0 + soc * (sto_n_1 - sto_n_0)
        curr_sto_p = sto_p_0 - soc * (sto_p_0 - sto_p_1)
        
        # OCV = U_p(sto_p) - U_n(sto_n)
        v = param.evaluate(U_p(pybamm.Scalar(curr_sto_p))) - \
            param.evaluate(U_n(pybamm.Scalar(curr_sto_n)))
        ocv_values.append(float(v))
    
    return interp1d(soc_range, ocv_values, kind='linear', fill_value="extrapolate")

# 初始化
get_ocv = get_chen2020_ocv_func()


# 加载 Chen2020 参数集
# param = pybamm.ParameterValues("Chen2020")

# # 打印所有键名，按字母顺序排列方便查看
# print("-" * 30)
# print("当前 Chen2020 所有可用参数键名：")
# for key in sorted(param.keys()):
#     print(key)
# print("-" * 30)