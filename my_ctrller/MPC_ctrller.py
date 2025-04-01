import numpy as np
from scipy.optimize import minimize
from .base import Controller, Action


class NMPCController(Controller):
    def __init__(self,
                 prediction_horizon=36,
                 glucose_target=108,
                 insulin_max=5.0,
                 alpha=1.0,
                 beta=0.1):
        # super().__init__()
        # 控制器参数
        self.N = prediction_horizon  # 预测步长（分钟）
        self.G_target = glucose_target  # 血糖目标值 (mg/dL)
        self.insulin_max = insulin_max  # 最大胰岛素输注速率 (U/h)
        self.alpha = alpha  # 血糖跟踪权重
        self.beta = beta  # 控制量权重

        # 模型参数（来自论文的Mansell模型）
        self.Td = 1.0  # 离散化时间步长（分钟）
        self.SI = 1e-4  # 胰岛素敏感性参数
        self.n_params = {  # 模型其他参数
            'k_x': 0.05, 'n_t': 0.08,
            'n_T': 0.1, 'n_C': 0.2,
            'k_1': 0.05, 'k_2': 0.1,
            'P_c': 0.01, 'V_p': 0.12
        }

    def discretize_model(self, x, u):
        """离散化状态方程（论文公式5-6）"""
        # 状态变量 x = [U_S, I, Q, P_S, G]
        # 控制输入 u = 胰岛素输注速率 (U/h)
        x_next = np.zeros_like(x)

        # 矩阵计算（简化实现）
        k_x, n_t, n_T, n_C, k_1, k_2, P_c, V_p = self.n_params.values()

        x_next[0] = x[0] + self.Td * (-k_x * x[0] + u)
        x_next[1] = x[1] + self.Td * (k_x * x[0] - (n_t + n_T) * x[1] + n_t * x[2])
        x_next[2] = x[2] + self.Td * (V_p * x[1] - (n_t + n_C) * x[2])
        x_next[3] = x[3] + self.Td * (-k_1 * x[3] + k_2 * x[4])
        x_next[4] = x[4] + self.Td * (-self.SI * x[4] * x[2] + P_c * (x[3] - x[4]))

        return x_next

    def cost_function(self, u_sequence, x0):
        """NMPC目标函数（论文公式9）"""
        total_cost = 0.0
        x = x0.copy()

        for k in range(self.N):
            u = u_sequence[k]
            # 状态预测
            x = self.discretize_model(x, u)
            # 血糖跟踪误差项
            G = x[4]
            total_cost += self.alpha * (G - self.G_target) ** 2
            # 控制量惩罚项
            total_cost += self.beta * u ** 2

        return total_cost

    def policy(self, observation, reward, done, **info):
        """NMPC控制策略"""
        # 获取当前血糖值（假设observation包含状态信息）
        current_state = np.array([
            observation.CGM,  # 假设CGM为第一个状态
            0.0, 0.0, 0.0,  # 其他状态需要估计（简化实现）
            observation.CGM  # 血糖值作为最后一个状态
        ])

        # 构造优化问题
        u_init = np.zeros(self.N)  # 初始猜测
        bounds = [(0, self.insulin_max)] * self.N  # 胰岛素非负且有上限

        # 求解非线性优化
        res = minimize(
            fun=self.cost_function,
            x0=u_init,
            args=(current_state,),
            bounds=bounds,
            method='SLSQP'
        )

        # 应用第一控制量
        optimal_insulin = res.x[0] if res.success else 0.0

        return Action(basal=optimal_insulin, bolus=0.0)

    def reset(self):
        """重置控制器状态"""
        pass  # NMPC无记忆状态