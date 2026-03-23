'''Python version of D2_ESN.'''

import torch
import numpy as np
import random
import cv2
import os
from .functions import *
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

seed = 11
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class ESN_2D(torch.nn.Module):
    def __init__(self, input_dim=1, n_reservoir=100, spectral_radius=(0.9, 0.9), alpha=5,
                 connectivity=0.1, noise_level=1e-4, start_node=(1, 1), activation=torch.tanh, device_override=None):
        super(ESN_2D, self).__init__()
        self._device = device_override if device_override is not None else device
        self.n_reservoir = n_reservoir  # Number of reservoir neurons
        self.spectral_radius = spectral_radius  # Spectral radius for the reservoir
        self.alpha = alpha  # Ridge regression coefficient
        self.connectivity = connectivity # 连接密度，实现了对水库内部连接的稀疏化。
        self.noise_level = noise_level  # 在状态更新中加入噪声，以增加模型的鲁棒性
        self.start_node = start_node
        self.activation = activation

        W_in = 2 * np.random.rand(input_dim, n_reservoir) - 1  # Input weights

        W_res_1 = np.random.rand(n_reservoir, n_reservoir) - 0.5  # Reservoir 内部权重
        mask = np.random.rand(n_reservoir, n_reservoir) < connectivity
        W_res_1 *= mask

        W_res_2 = np.random.rand(n_reservoir, n_reservoir) - 0.5  # Reservoir 内部权重
        mask = np.random.rand(n_reservoir, n_reservoir) < connectivity
        W_res_2 *= mask

        # Ensure the reservoir has the desired spectral radius
        rho_W_res = max(abs(np.linalg.eig(W_res_1)[0]))  # 谱半径最大特征值的绝对值
        W_res_1 *= self.spectral_radius[0] / rho_W_res

        rho_W_res = max(abs(np.linalg.eig(W_res_2)[0]))  # 谱半径最大特征值的绝对值
        W_res_2 *= self.spectral_radius[1] / rho_W_res

        # Convert weights to PyTorch tensors
        self.W_in = torch.nn.Parameter(torch.from_numpy(W_in).float(), requires_grad=False).to(self._device)
        self.W_res_1 = torch.nn.Parameter(torch.from_numpy(W_res_1).float(), requires_grad=False).to(self._device)
        self.W_res_2 = torch.nn.Parameter(torch.from_numpy(W_res_2).float(), requires_grad=False).to(self._device)

    # 更新水库状态
    def update_reservoir(self, x, state1, state2):
        """
        x: [batch_size]
        state: [batch_size, n_reservoir]
        """
        # 避免在最内层循环里重复触发 .to(device) 的包装/拷贝。
        if x.device != self._device:
            x = x.to(self._device)
        pre_activation = torch.mm(x.unsqueeze(1), self.W_in) + torch.mm(state1, self.W_res_1) + torch.mm(state2, self.W_res_2)
        state_new = self.activation(pre_activation)
        return state_new
    
    # 自定义岭回归（论文 Eq.(3)）
    # 论文要求特征 f = [W_out, b]，其中 b 为 bias
    # 需要在 states 上增广一列 1 以吸收 bias 项
    def ridge_regression(self, states, targets):
        """
        论文 Eq.(3): [W_out, b]^T = (H̃^T H̃ + λ²I)^{-1} H̃^T U
        其中 H̃ 是增广的隐藏状态矩阵（增广一列 1）
        
        Parameters:
        -----------
        states : [batch, length, dim]  (dim = 2*n_reservoir)
        targets : [batch, length]
        
        Returns:
        --------
        feature : [batch, dim+1]  (W_out 和 b 的拼接，作为动态特征)
        """
        batch, length, dim = states.shape
        
        # 增广一列 1 以吸收 bias（论文 Eq.(3) 的 H̃）
        ones = states.new_ones((batch, length, 1))
        states_aug = torch.cat([states, ones], dim=2)  # [batch, length, dim+1]
        
        targets = targets.unsqueeze(2)  # [batch, length, 1]
        
        # 正则化系数 λ²（论文 Eq.(3) 中为 λ²I）
        dim_aug = dim + 1
        # NOTE: 为避免每次重复分配 eye + repeat，这里缓存 [1, dim_aug, dim_aug] 的单位阵底座。
        # 仅做广播扩展，不改变数值计算逻辑（仍然是 HtH + λ²I）。
        if (not hasattr(self, "_ridge_I_base")) or (self._ridge_I_base is None) or (int(self._ridge_I_base.shape[-1]) != int(dim_aug)):
            self._ridge_I_base = torch.eye(dim_aug, dtype=states.dtype, device=states.device).unsqueeze(0)
        I = self._ridge_I_base.expand(batch, -1, -1) * (self.alpha ** 2)  # 论文用 λ²
        
        # 岭回归求解 (H̃^T H̃ + λ²I)^{-1} H̃^T U
        HtH = torch.bmm(states_aug.transpose(1, 2), states_aug)
        HtH_plus_lambdaI = HtH + I
        HtU = torch.bmm(states_aug.transpose(1, 2), targets)

        solver_env = (os.environ.get("RES_SAM_RIDGE_SOLVER", "") or "").strip().lower()
        solver = solver_env
        if not solver:
            solver = "solve"

        if solver == "solve":
            # A @ X = B  -> X = solve(A, B)
            W_out_with_bias = torch.linalg.solve(HtH_plus_lambdaI, HtU)  # [batch, dim+1, 1]
        else:
            HtH_plus_lambdaI_inv = torch.inverse(HtH_plus_lambdaI)
            W_out_with_bias = torch.bmm(HtH_plus_lambdaI_inv, HtU)  # [batch, dim+1, 1]
        
        # 返回 [W_out, b] 作为特征（论文 Eq.(3) 后的描述）
        # squeeze 后形状为 [batch, dim+1]
        return W_out_with_bias.squeeze(2)

    def forward(self, input_image):
        """
        论文 Eq.(2)-(3)：2D-ESN 拟合，返回动态特征 f = [W_out, b]
        
        Parameters:
        -----------
        input_image : [batch_size, height, width]
            输入 patch
        
        Returns:
        --------
        feature : [batch_size, 2*n_reservoir + 1]
            动态特征 f = [W_out, b]，其中 W_out 维度为 2*n_reservoir，b 为标量
        """
        # 避免每次无条件 detach()/to(device) 触发额外的 tensor 包装/拷贝。
        # 在 no_grad 推理下不需要 detach；仅在 device/dtype 不匹配时才转换。
        if input_image.device != self._device or input_image.dtype != torch.float32:
            input_image = input_image.to(device=self._device, dtype=torch.float32)
        batch, height, width = input_image.shape
        initial_state = torch.zeros((batch, self.n_reservoir), dtype=torch.float32, device=self._device)

        prev_row = torch.zeros((batch, width, self.n_reservoir), dtype=torch.float32, device=self._device)
        curr_row = torch.zeros((batch, width, self.n_reservoir), dtype=torch.float32, device=self._device)

        curr_row[:, 0, :] = self.update_reservoir(input_image[:, 0, 0], initial_state, initial_state)  # init
        for t in range(1, width):
            curr_row[:, t, :] = self.update_reservoir(input_image[:, 0, t], curr_row[:, t - 1, :], initial_state)

        start_h = int(self.start_node[0])
        start_t = int(self.start_node[1])
        pre_states_chunks = []
        targets_chunks = []

        for h in range(1, height):
            prev_row, curr_row = curr_row, prev_row
            curr_row.zero_()
            curr_row[:, 0, :] = self.update_reservoir(input_image[:, h, 0], initial_state, prev_row[:, 0, :])
            for t in range(1, width):
                curr_row[:, t, :] = self.update_reservoir(
                    input_image[:, h, t],
                    curr_row[:, t - 1, :],
                    prev_row[:, t, :],
                )

            if h >= start_h and start_t < width:
                if start_t <= 0:
                    continue
                left_states = curr_row[:, (start_t - 1):(width - 1), :]
                up_states = prev_row[:, start_t:width, :]
                if left_states.shape[1] != up_states.shape[1] or left_states.shape[1] <= 0:
                    continue
                pre_states_chunks.append(torch.cat([left_states, up_states], dim=2))
                targets_chunks.append(input_image[:, h, start_t:width])

        if not pre_states_chunks:
            pre_states = torch.zeros((batch, 0, 2 * self.n_reservoir), dtype=torch.float32, device=self._device)
            targets = torch.zeros((batch, 0), dtype=torch.float32, device=self._device)
        else:
            pre_states = torch.cat(pre_states_chunks, dim=1)
            targets = torch.cat(targets_chunks, dim=1)

        return self.ridge_regression(pre_states, targets)


def load_and_preprocess_image(file_path,):
    jpg_data = cv2.imread(file_path)[:, :, 0].astype(np.float32)  # 其实是取了图像的第0维
    height, width = jpg_data.shape
    jpg_data = cv2.resize(jpg_data, (height//4, width//4))  # 放缩成1/2的大小 注意要整除用“//”号
    return jpg_data


def normalize(image, normpr):
    """
    :param image: 原始图像数据，numpy数组格式
    :return: 归一化后的图像数据
    """
    # 将图像数据转换为float类型，以便进行除法操作
    image = image.astype(np.float32)
    # 图像归一化
    """这里不用除，放缩了图像会导致x过小（参考状态转移方程），进而ens输入学不到东西
    到底是第0维上取平均还是第1维上取平均还需进一步考虑"""
    std = np.std(image) * normpr
    normalized_image = (image - np.mean(image, axis=0)) / std   # (image - np.min(image))  / (np.max(image) - np.min(image))
    return normalized_image
