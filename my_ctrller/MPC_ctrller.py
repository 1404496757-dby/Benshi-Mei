import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Cloud:
    """数据云类，存储每个云的信息"""
    mu: np.ndarray  # 均值
    sigma: float  # 均方长度
    M: int  # 数据点数量
    k_add: int  # 添加时间步
    P: float  # 比例增益
    I: float  # 积分增益
    D: float  # 微分增益
    R: float  # 操作点补偿


class RECCoController:
    """Python 实现的 RECCo 控制器"""

    def __init__(self, u_min: float, u_max: float, y_min: float, y_max: float,
                 Ts: float, tau: float, G_sign: int):
        """
        初始化 RECCo 控制器

        参数:
            u_min: 控制信号最小值
            u_max: 控制信号最大值
            y_min: 输出信号最小值
            y_max: 输出信号最大值
            Ts: 采样时间(秒)
            tau: 估计的时间常数(秒)
            G_sign: 过程增益的符号(1或-1)
        """
        # 控制器参数
        self.u_min = u_min
        self.u_max = u_max
        self.y_min = y_min
        self.y_max = y_max
        self.Ts = Ts
        self.tau = tau
        self.G_sign = G_sign

        # 默认参数
        self.gamma_max = 0.93
        self.n_add = 20
        self.d_dead = 0.01
        self.sigma_L = 1e-6

        # 调整适应增益
        scale = (u_max - u_min) / 20
        self.alpha_P = 0.1 * scale
        self.alpha_I = 0.1 * scale
        self.alpha_D = 0.1 * scale
        self.alpha_R = 0.1 * scale

        # 状态变量
        self.clouds: List[Cloud] = []
        self.last_add_k = -np.inf
        self.e_prev = 0.0
        self.Sigma_e = 0.0
        self.y_r_prev = 0.0
        self.k = 0

    def control(self, r: float, y: float) -> tuple:
        """
        计算控制信号

        参数:
            r: 当前参考信号
            y: 当前过程输出

        返回:
            (u, y_r): 控制信号和参考模型输出
        """
        self.k += 1

        # 1. 参考模型
        a_r = 1 - self.Ts / self.tau
        y_r = a_r * self.y_r_prev + (1 - a_r) * r
        self.y_r_prev = y_r

        # 计算跟踪误差
        e = y_r - y
        Delta_e = e - self.e_prev
        self.e_prev = e

        # 更新误差积分(带抗饱和)
        u_prev = self._compute_control(e, Delta_e, y_r, y)
        if u_prev > self.u_min and u_prev < self.u_max:
            self.Sigma_e += e

        # 2. 演化法则
        x = self._normalize_input(e, y_r, y)

        if not self.clouds:
            # 第一个数据点 - 创建第一个云
            self._add_new_cloud(x, [0, 0, 0, 0])
        else:
            # 计算与所有云的关联度
            lambda_, gamma, active_cloud = self._compute_association(x)

            # 检查是否需要添加新云
            if max(gamma) < self.gamma_max and (self.k - self.last_add_k) >= self.n_add:
                # 计算新云的初始参数
                if len(self.clouds) == 1:
                    theta_new = [self.clouds[0].P, self.clouds[0].I,
                                 self.clouds[0].D, self.clouds[0].R]
                else:
                    theta_new = np.zeros(4)
                    for i, cloud in enumerate(self.clouds):
                        theta_new += lambda_[i] * np.array([cloud.P, cloud.I, cloud.D, cloud.R])

                self._add_new_cloud(x, theta_new)
                active_cloud = len(self.clouds) - 1

            # 3. 适应法则 - 仅更新活跃云
            if active_cloud is not None:
                self._adapt_parameters(active_cloud, e, Delta_e, r, y)

        # 计算控制信号
        u = self._compute_control(e, Delta_e, y_r, y)

        return u, y_r

    def _normalize_input(self, e: float, y_r: float, y: float) -> np.ndarray:
        """归一化输入向量"""
        Delta_y = self.y_max - self.y_min
        Delta_e = Delta_y / 2
        return np.array([e / Delta_e, (y_r - self.y_min) / Delta_y])

    def _compute_association(self, x: np.ndarray) -> tuple:
        """计算与所有云的关联度"""
        gamma = np.zeros(len(self.clouds))

        for i, cloud in enumerate(self.clouds):
            gamma[i] = 1 / (1 + np.linalg.norm(x - cloud.mu) ** 2 + cloud.sigma - np.linalg.norm(cloud.mu) ** 2)

        lambda_ = gamma / np.sum(gamma)
        active_cloud = np.argmax(gamma) if len(gamma) > 0 else None

        return lambda_, gamma, active_cloud

    def _add_new_cloud(self, x: np.ndarray, theta: list):
        """添加新云"""
        new_cloud = Cloud(
            mu=x,
            sigma=np.linalg.norm(x) ** 2,
            M=1,
            k_add=self.k,
            P=theta[0],
            I=theta[1],
            D=theta[2],
            R=theta[3]
        )

        self.clouds.append(new_cloud)
        self.last_add_k = self.k
        print(f'Added new cloud at k={self.k}. Total clouds: {len(self.clouds)}')

    def _adapt_parameters(self, cloud_idx: int, e: float, Delta_e: float, r: float, y: float):
        """参数适应法则"""
        cloud = self.clouds[cloud_idx]
        e_p = r - y  # 过程误差

        # 计算参数变化
        denom = 1 + r ** 2

        if self.k * self.Ts < 5 * self.tau:  # 前5个时间常数使用绝对值
            delta_P = self.alpha_P * self.G_sign * (abs(e_p * e)) / denom
            delta_I = self.alpha_I * self.G_sign * (abs(e_p * self.Sigma_e)) / denom
            delta_D = self.alpha_D * self.G_sign * (abs(e_p * Delta_e)) / denom
        else:
            delta_P = self.alpha_P * self.G_sign * (e_p * e) / denom
            delta_I = self.alpha_I * self.G_sign * (e_p * self.Sigma_e) / denom
            delta_D = self.alpha_D * self.G_sign * (e_p * Delta_e) / denom

        delta_R = self.alpha_R * self.G_sign * (e) / denom

        # 死区机制
        if abs(e) < self.d_dead:
            delta_P = delta_I = delta_D = delta_R = 0
        # 应用泄漏
        cloud.P = (1 - self.sigma_L) * cloud.P + delta_P
        cloud.I = (1 - self.sigma_L) * cloud.I + delta_I
        cloud.D = (1 - self.sigma_L) * cloud.D + delta_D
        cloud.R = (1 - self.sigma_L) * cloud.R + delta_R
        # 参数投影 (确保P,I,D >= 0)
        cloud.P = max(0, cloud.P)
        cloud.I = max(0, cloud.I)
        cloud.D = max(0, cloud.D)
        # 更新云统计信息
        cloud.M += 1
        cloud.mu = ((cloud.M - 1) / cloud.M) * cloud.mu + (1 / cloud.M) * np.array([e, y])
        cloud.sigma = ((cloud.M - 1) / cloud.M) * cloud.sigma + (1 / cloud.M) * (e ** 2 + y ** 2)

    def _compute_control(self, e: float, Delta_e: float, y_r: float, y: float) -> float:
        """计算控制信号"""
        if not self.clouds:
            return 0.0

        x = self._normalize_input(e, y_r, y)
        lambda_, _, _ = self._compute_association(x)

        u_total = 0.0
        for i, cloud in enumerate(self.clouds):
            u_i = cloud.P * e + cloud.I * self.Sigma_e + cloud.D * Delta_e + cloud.R
            u_total += lambda_[i] * u_i

        # 限制控制信号
        return np.clip(u_total, self.u_min, self.u_max)