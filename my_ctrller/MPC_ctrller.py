from .base import Controller, Action
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pkg_resources
import logging

logger = logging.getLogger(__name__)
CONTROL_QUEST = pkg_resources.resource_filename('simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename('simglucose', 'params/vpatient_params.csv')


class NMPCController(Controller):
    """Nonlinear MPC Controller for blood glucose regulation"""

    def __init__(self,
                 prediction_horizon=36,
                 glucose_target=108,
                 insulin_max=5.0,
                 alpha=1.0,
                 beta=0.1):
        # 加载患者参数文件
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)

        # 控制器参数
        self.N = prediction_horizon  # 预测步长（分钟）
        self.G_target = glucose_target  # 目标血糖值 (mg/dL)
        self.insulin_max = insulin_max  # 最大胰岛素速率 (U/min)
        self.alpha = alpha  # 血糖跟踪权重
        self.beta = beta  # 控制量权重

        # 状态估计缓存
        self.estimated_states = np.zeros(5)  # [U_S, I, Q, P_S, G]

    def discretize_model(self, x, u, params):
        """Mansell模型离散化（基于论文公式5-6）"""
        x_next = np.zeros_like(x)
        k_x = params.get('k_x', 0.05)
        n_t = params.get('n_t', 0.08)
        n_T = params.get('n_T', 0.1)
        n_C = params.get('n_C', 0.2)
        k_1 = params.get('k_1', 0.05)
        k_2 = params.get('k_2', 0.1)
        P_c = params.get('P_c', 0.01)
        V_p = params.get('V_p', 0.12)
        SI = params.get('SI', 1e-4)

        # 状态方程离散化
        x_next[0] = x[0] + (-k_x * x[0] + u)  # U_S
        x_next[1] = x[1] + (k_x * x[0] - (n_t + n_T) * x[1] + n_t * x[2]) # I
        x_next[2] = x[2] + (V_p * x[1] - (n_t + n_C)*x[2])
        x_next[3] = x[3] + (-k_1 * x[3] + k_2 * x[4])  # P_S
        x_next[4] = x[4] + (-SI * x[4] * x[2] + P_c * (x[3] - x[4]))  # G
        return x_next

    def cost_function(self, u_sequence, x0, params):
        """NMPC目标函数（论文公式9）"""
        total_cost = 0.0
        x = x0.copy()
        for k in range(self.N):
            x = self.discretize_model(x, u_sequence[k], params)
            total_cost += self.alpha * (x[4] - self.G_target) ** 2  # 血糖跟踪项
            total_cost += self.beta * u_sequence[k] ** 2  # 控制量惩罚项
        return total_cost

    def policy(self, observation, reward, done, **kwargs):
        """NMPC控制策略"""
        # 获取患者参数
        pname = kwargs.get('patient_name')
        sample_time = kwargs.get('sample_time', 1)
        params = self._get_patient_params(pname)

        # 状态更新（假设CGM为第五个状态）
        self.estimated_states[4] = observation.CGM  # 更新血糖观测值

        # 求解优化问题
        bounds = [(0, self.insulin_max)] * self.N
        res = minimize(
            self.cost_function,
            x0=np.zeros(self.N),
            args=(self.estimated_states, params),
            bounds=bounds,
            method='SLSQP'
        )

        # 应用最优控制量
        basal = res.x[0] if res.success else 0.0
        return Action(basal=basal, bolus=0.0)

    def _get_patient_params(self, name):
        """获取患者参数（兼容原始参数文件和Mansell模型参数）"""
        params = {}

        # 1. 从原始参数文件获取基础参数
        if any(self.patient_params.Name.str.match(name)):
            patient_data = self.patient_params[self.patient_params.Name.str.match(name)]
            params.update({
                'BW': patient_data.BW.values[0],
                'u2ss': patient_data.u2ss.values[0],
                # 添加其他原始参数...
            })

        # 2. 硬编码Mansell模型参数（实际应用时应从单独文件加载）
        params.update({
            'k_x': 0.05, 'n_t': 0.08, 'n_T': 0.1, 'n_C': 0.2,
            'k_1': 0.05, 'k_2': 0.1, 'P_c': 0.01, 'V_p': 0.12, 'SI': 1e-4
        })

        return params

    def reset(self):
        """重置状态估计"""
        self.estimated_states = np.zeros(5)