import numpy as np
from collections import namedtuple
from scipy.optimize import minimize

# 保持与my_ctrller相同的Action结构
Action = namedtuple('Action', ['basal', 'bolus'])


class MPCController:
    def __init__(self, patient_model=None, prediction_horizon=10, control_horizon=5):
        """
        MPC控制器初始化

        参数：
        - patient_model: 患者模型函数 f(x,u) -> x_next
        - prediction_horizon: 预测步长
        - control_horizon: 控制步长
        """
        self.patient_model = patient_model or self._default_model
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon

        # 控制参数
        self.target_bg = 100.0  # 目标血糖(mg/dL)
        self.max_insulin = 5.0  # 最大单次胰岛素剂量(U)
        self.min_insulin = 0.0  # 最小胰岛素剂量

        # 状态初始化
        self.last_bg = None
        self.insulin_history = []
        self.bg_history = []

        # 权重矩阵 (Q:状态误差, R:控制量)
        self.Q = np.diag([1.0])  # 血糖误差权重
        self.R = np.diag([0.1])  # 胰岛素变化量权重

    def policy(self, observation, reward, done, **info):
        """
        核心控制策略（与my_ctrller接口保持一致）
        """
        current_bg = observation.CGM
        self.last_bg = current_bg
        self.bg_history.append(current_bg)

        # 调用MPC计算最优控制量
        optimal_insulin = self._solve_mpc(current_bg)

        # 记录胰岛素历史
        self.insulin_history.append(optimal_insulin)

        # 返回Action对象（保持与原始格式一致）
        return Action(basal=optimal_insulin, bolus=0)

    def _solve_mpc(self, current_bg):
        """
        求解MPC优化问题
        """
        # 初始猜测（使用上次控制量或零）
        u0 = [0.0] * self.control_horizon
        if len(self.insulin_history) > 0:
            u0 = [self.insulin_history[-1]] * self.control_horizon

        # 定义优化问题边界
        bounds = [(self.min_insulin, self.max_insulin)] * self.control_horizon

        # 求解优化问题
        result = minimize(
            fun=self._cost_function,
            x0=u0,
            args=(current_bg,),
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 50, 'ftol': 1e-6}
        )

        # 返回第一个控制量（MPC只执行第一步）
        return float(result.x[0])

    def _cost_function(self, u, current_bg):
        """
        MPC代价函数计算
        """
        # 扩展控制序列到预测时域
        full_u = list(u) + [u[-1]] * (self.prediction_horizon - len(u))

        # 模拟预测时域内的系统行为
        x = current_bg
        cost = 0.0

        for k in range(self.prediction_horizon):
            # 系统状态更新（使用患者模型）
            x_next = self.patient_model(x, full_u[k])

            # 计算代价：血糖误差 + 胰岛素变化惩罚
            bg_error = x_next - self.target_bg

            # 胰岛素变化量（考虑历史）
            if k == 0 and len(self.insulin_history) > 0:
                delta_u = full_u[k] - self.insulin_history[-1]
            elif k > 0:
                delta_u = full_u[k] - full_u[k - 1]
            else:
                delta_u = 0.0

            # 累计代价
            cost += bg_error ** 2 * self.Q[0, 0] + delta_u ** 2 * self.R[0, 0]

            x = x_next

        return cost

    def _default_model(self, bg, insulin):
        """
        默认患者模型（论文中的Sorensen模型简化版）
        """
        # 简化的血糖-胰岛素动力学模型
        # 实际应用中应替换为论文中的完整模型
        delta_bg = -0.5 * bg - 0.8 * insulin + 50
        return bg + delta_bg * 0.1  # 离散时间步长

    def reset(self):
        """
        重置控制器状态
        """
        self.last_bg = None
        self.insulin_history = []
        self.bg_history = []


# 保持与my_ctrller相同的使用接口
# if __name__ == "__main__":
#     # 初始化控制器
#     mpc = MPCController()
#
#
#     # 模拟血糖观测值
#     class Observation:
#         def __init__(self, cgm):
#             self.CGM = cgm
#
#
#     # 测试控制循环
#     for bg in [180, 170, 160, 150, 140, 130, 120, 110, 105, 100]:
#         obs = Observation(bg)
#         action = mpc.policy(obs, reward=0, done=False)
#         print(f"血糖: {bg} mg/dL | 胰岛素剂量: {action.basal:.2f} U")