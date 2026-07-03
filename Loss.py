import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

def smoothloss(y_pred):
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :])
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :])
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1])
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0
#相似性loss函数

def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]
    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])
    Jdet = Jdet0 - Jdet1 + Jdet2
    return Jdet
#计算三维形变场的 Jacobian determinant（雅可比行列式）

def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)
#计算均方误差 (Mean Squared Error, MSE)。


def msewithmask(y_true, y_pred, mask):
    return torch.mean(((y_true - y_pred)*mask) ** 2)
# 带掩码的均方误差（Masked Mean Squared Error, MSE）

def neg_Jdet_loss(y_pred, sample_grid):
    """
    参数:
        y_pred : torch.Tensor
            网络预测的位移场 (deformation field)，形状 [B, H, W, D, 3]。
        sample_grid : torch.Tensor
            原始坐标网格 (identity grid)，形状同 y_pred。
    返回:
        torch.Tensor
            标量损失值（float），越小表明形变越平滑、无折叠。
    """
    # 计算 Jacobian determinant (体积变化比例)
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    # 只选取那些 determinant < 0 的部分（负值表示局部空间折叠）
    selected_neg_Jdet = F.relu(neg_Jdet)
    # 求平均作为损失值
    return torch.mean(selected_neg_Jdet)
#计算负 Jacobian determinant 损失，用于惩罚形变场中的折叠 (folding)。（正则化损失）
    
def dice(y_true, y_pred):
    """
    参数:
        y_true : torch.Tensor
            真实标注（通常为 0/1 或二值掩码）。
        y_pred : torch.Tensor
            模型输出的预测概率或分割结果。
    返回:
        torch.Tensor
            标量 Dice 损失值（负的 Dice 系数）。
            返回负值是因为在训练中要最小化 loss，
            而 Dice 系数本身是希望越大越好。
    """
    ndims = len(list(y_pred.size())) - 2                 # 通道数以后剩下的维度数
    vol_axes = list(range(2, ndims + 2))                # 空间维度求和轴
    # Dice 系数的分子：2 * |A ∩ B|
    top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    # 分母：|A| + |B|，加一个小常数防止除以 0
    bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom)
    return -dice     # 负号是为了作为损失最小化（优化器会最小化）
#计算 Dice 损失（用于二值或概率分割任务）。

def magnitude_loss(flow_1, flow_2):
    """
    参数:
        flow_1 : torch.Tensor
            第一个流场张量 (例如 forward flow)
        flow_2 : torch.Tensor
            第二个流场张量 (例如 backward flow)
    返回:
        torch.Tensor
            标量损失值（两个流场整体 magnitude 差异）
    """
    num_ele = torch.numel(flow_1)
    flow_1_mag = torch.sum(torch.abs(flow_1))
    flow_2_mag = torch.sum(torch.abs(flow_2))
    diff = (torch.abs(flow_1_mag - flow_2_mag)) / num_ele
    return diff
#计算两个形变(流场)整体强度的差异，用于约束双向流场或多阶段一致性。（正则化损失）

def cc_loss(x, y):
    # 根据互相关公式进行计算
    dim = [2, 3, 4]
    mean_x = torch.mean(x, dim, keepdim=True)
    mean_y = torch.mean(y, dim, keepdim=True)
    mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
    mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
    stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
    stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
    return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))
#全局互相关损失函数

def ncc_loss(I, J, win=None, mind=False):
    if win is None:
        win = [9] * 3
    device = I.device  
    sum_filter = torch.ones(1, 1, *win).to(device)
    #sum_filter = torch.ones(1, I.shape[1], *win).to("cuda")
    pad = math.floor(win[0] / 2)
    stride = (1, 1, 1)
    pading = (pad, pad, pad)
    I_var, J_var, cross = compute_local_sums(I, J, sum_filter, stride, pading, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)
#局部NCC损失函数

def compute_local_sums(I, J, filter, stride, padding, win):
    # 局部平方和 & 乘积项
    I2 = I * I
    J2 = J * J
    IJ = I * J
    # 用全 1 卷积核计算局部求和
    I_sum  = F.conv3d(I,  filter, stride=stride, padding=padding)
    J_sum  = F.conv3d(J,  filter, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filter, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filter, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filter, stride=stride, padding=padding)
    # 局部窗口体积
    win_size = np.prod(win) * I.shape[1]
    # 局部均值
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    # 局部协方差项
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    # 局部方差项
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross
#局部 NCC (Local Normalized Cross-Correlation) 的关键辅助函数 ——它负责在局部窗口内高效计算各类统计量，用于后续的互相关损失


class NCC(torch.nn.Module):

    # local (over window) normalized cross correlation

    def __init__(self, win=5, eps=1e-5):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)
#局部归一化互相关损失 (Local Normalized Cross-Correlation, LNCC)
            
class multi_window_loss(torch.nn.Module):
    """
    功能：
        该模块用于图像配准任务中计算多窗口（多尺度）的局部 NCC 相似性损失。
        每个尺度对应一个不同的滑动窗口大小，通过融合不同尺度信息，
        可同时关注图像的全局结构与局部细节，从而提升配准的鲁棒性与精度。
    参数说明：
        win   : list[int]，各尺度对应的窗口大小，如 [11, 9, 7]
        eps   : float，防止除零的微小常数
        gamma : float，权重衰减系数，用于对不同尺度的损失加权（通常 <1）

    返回值：
        total_NCC : list[Tensor]，包含每个尺度的 NCC 损失值，可根据需要求和或加权平均。
                    在实际训练中通常将这些损失求和得到最终的相似性损失。
    """
    def __init__(self, win=[11, 9, 7], eps=1e-5, gamma=0.5):
        super(multi_window_loss, self).__init__()
        self.num_scale = len(win)    # 多尺度数量
        self.gamma = gamma           # 权重衰减系数
        self.similarity_metric = []  # 存放各尺度 NCC 模块
        # 为每个尺度创建一个 NCC 相似性模块
        for i in range(self.num_scale):
            self.similarity_metric.append(NCC(win=win[i]))

    def forward(self, I, J):
        """
        前向传播：
        输入两幅图像 I, J（通常是 warped_moving 和 fixed 图像），
        分别在不同窗口尺度下计算局部 NCC 损失，并给予尺度权重。
        参数：
            I : Tensor，输入图像（例如变换后的 moving 图像）
            J : Tensor，目标图像（例如 fixed 图像）
        返回：
            total_NCC : list[Tensor]，各尺度的 NCC 损失值
        """
        total_NCC = []   # 保存所有尺度下的 NCC 损失

        for i in range(self.num_scale):
            # 当前尺度的 NCC 计算
            current_NCC = self.similarity_metric[i](I, J)

            # 按 γ^i 加权（高尺度权重更高或更低取决于 gamma）
            total_NCC.append(current_NCC * (self.gamma ** i))
        return total_NCC
#多尺度局部归一化互相关损失（Multi-scale Local Normalized Cross-Correlation Loss）

class multi_resolution_NCC(torch.nn.Module):
    """
    功能：
        在医学图像配准任务中，不同分辨率可以捕捉不同尺度下的相似性信息：
            - 高分辨率：保留细节，适合边界对齐与局部纹理
            - 低分辨率：保留全局结构，适合初步对齐
        通过逐层下采样（avg_pool3d）构建分辨率金字塔，并在每个分辨率上计算 NCC，相加得到最终损失。
        这种方法对大形变和多尺度结构变化更鲁棒。
    参数：
        win   : int，初始计算 NCC 的局部窗口边长（例如 11）
                多分辨率中，每降低一个分辨率，窗口会减小 (win - i*2)
        eps   : float，防止除零的微小常数
        scale : int，多分辨率的层数（>=1）。scale=3 表示原图 + 两次下采样
    返回值：
        一个标量损失值（所有分辨率 NCC 的加权和）
    """
    def __init__(self, win=None, eps=1e-5, scale=3):
        super(multi_resolution_NCC, self).__init__()
        self.num_scale = scale
        self.similarity_metric = []  # 存放不同分辨率下的 NCC

        # 为每个分辨率创建一个 NCC 模块，窗口大小依次减小
        for i in range(scale):
            self.similarity_metric.append(NCC(win=win - (i * 2), eps=eps))
    def forward(self, I, J):
        """
        前向计算：
        1. 在原始分辨率计算 NCC
        2. 下采样到低一层分辨率，重复计算，直到达到 num_scale 层
        3. 对各分辨率 NCC 值加权求和，作为最终损失值

        参数：
            I : Tensor，输入图像（如变换后的 moving 图像），维度 [B, C, D, H, W]
            J : Tensor，目标图像（如 fixed 图像），维度与 I 相同

        返回：
            标量损失值（float tensor）
        """
        total_NCC = []

        for i in range(self.num_scale):
            # 当前分辨率 NCC
            current_NCC = self.similarity_metric[i](I, J)

            # 加权（低分辨率权重更小）
            total_NCC.append(current_NCC / (2 ** i))

            # 下采样到下一分辨率（平均池化），stride=2 缩小一半
            I = nn.functional.avg_pool3d(
                I, kernel_size=3, stride=2, padding=1,
                count_include_pad=False
            )
            J = nn.functional.avg_pool3d(
                J, kernel_size=3, stride=2, padding=1,
                count_include_pad=False
            )
        # 加和得到最终损失（标量）
        return sum(total_NCC)
#该模块实现了多分辨率（图像金字塔）版本的局部归一化互相关损失（NCC）。


def dicegup(im1, atlas):
    """
    --------------------------------------------------------------
    功能：
        该函数用于计算两张多类别分割图（如预测分割结果 im1 与真实标签 atlas）
        之间的平均 Dice 系数。Dice 系数反映了预测与标签的一致程度，
        取值范围 [0,1]，数值越大表示重叠越高、分割越准确。

    参数：
        im1   : Tensor 或 ndarray，预测结果（分割图或注册后标签图）
        atlas : Tensor 或 ndarray，真实标签（或模板标签图）

    返回：
        平均 Dice 系数（float）
        若两个输入中所有有效类别都为空，则返回 0

    计算方式：
        对于每一个存在于 atlas 中的类别 i：
            1. 计算 im1 和 atlas 中属于类别 i 的体素数 t1, t2
            2. 计算交集 voxel 数：((im1 == i) & (atlas == i)).sum()
            3. dice_i = 2 * intersection / (t1 + t2)
        最后对所有有效类别取平均。

    注意：
        - 背景类（label==0）不参与计算。
        - 如果某一类在其中一张图中不存在（t1==0或t2==0），跳过。
        - 结果是所有类别 Dice 的平均。
    --------------------------------------------------------------
    """
    unique_class = atlas.unique()  # 获取 atlas 中的所有类别标签值
    ret = 0                        # Dice 累计和
    num_count = 0                  # 有效类别计数（用于取平均）

    # 遍历每一个类别
    for i in unique_class:
        t1 = (im1 == i).sum()      # im1 中类别 i 的体素数量
        t2 = (atlas == i).sum()    # atlas 中类别 i 的体素数量

        # 跳过背景 (i == 0) 或该类缺失的情况
        if i == 0 or t1 == 0 or t2 == 0:
            continue

        # 计算该类的 Dice 系数：2 * |交集| / (|A| + |B|)
        ret += (atlas[im1 == i] == i).sum() * 2.0 / (t1 + t2)
        num_count += 1

    # 若没有有效类别，返回 0；否则返回平均 Dice
    if num_count == 0:
        return ret
    else:
        return ret / num_count
#计算多类别 Dice 相似系数（Generalized or Average Dice Coefficient）


def compute_per_channel_dice(input, target, classes=27, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    # assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    if input.size() != target.size():
        target = mask_to_one_hot(target, n_classes=classes)

    # input = flatten(input)
    # target = flatten(target)
    # target = target.float()\

    input = input.contiguous().view(input.size()[1], -1)
    target = target.contiguous().view(target.size()[1], -1).float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    dice_score = 2 * (intersect / denominator.clamp(min=epsilon))
    
    # 跳过背景通道（通道0），只计算前景类别的Dice
    return -dice_score[1:].mean()
#多类别 Dice Loss（基于概率 soft output）计算函数，

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)
#将多维输入重新排列成“通道优先的二维张量”

def mask_to_one_hot(mask, n_classes):
    """
    Convert a segmentation mask to one-hot coded tensor
    :param mask: mask tensor of size Bx1xDxMxN
    :param n_classes: number of classes
    :return: one_hot: BxCxDxMxN
    """
    one_hot_shape = list(mask.shape)
    one_hot_shape[1] = n_classes

    mask_one_hot = torch.zeros(one_hot_shape).to(mask.device)

    mask_one_hot.scatter_(1, mask.long(), 1)

    return mask_one_hot
#将分割掩膜 (mask) 转换为 one-hot 编码格式。


class WeightedCrossEntropyLoss(torch.nn.Module):
    """
    功能：
        动态地为每个类别分配权重，从而减轻类别不平衡对训练的影响。
        权重由当前批次预测的概率图计算得到：较小预测概率的类别会获得更大权重，
        促进网络学习那些难预测或数量较少的类别。
    原理公式（权重定义）：
        weight_c = (1 - p_c) / p_c
        其中 p_c 表示类别 c 的平均预测概率。
    参数：
        ignore_index : int, 默认值 -1  
            在计算损失时忽略的标签索引（例如背景类或填充值）。

    输入：
        input : torch.Tensor  
            网络预测输出，形状为 [N, C, D, H, W] 或 [N, C, H, W]，
            未经过 softmax（内部会自动计算 softmax）。

        target : torch.Tensor  
            真实标签（类别索引），形状为 [N, D, H, W] 或 [N, H, W]。
    返回：
        loss : torch.Tensor  
            加权交叉熵损失的标量。
    ---------------------------------------------------------------------------
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # 根据当前预测动态计算类别权重
        weight = self._class_weights(input)

        # 计算带权重的交叉熵损失
        return torch.nn.functional.cross_entropy(
            input, target, weight=weight, ignore_index=self.ignore_index
        )

    @staticmethod
    def _class_weights(input):
        """
        动态计算类别权重。
        原理：
            根据输入预测的平均概率分布，计算每类的相对权重，
            让预测较少的类别获得更大的权重。

        步骤：
            1️⃣ 对每个类做 softmax，得到概率图。
            2️⃣ 使用 flatten() 打平为 [C, N*D*H*W]。
            3️⃣ 计算权重：weight = (1 - mean_p) / mean_p。
        """

        # 转为概率分布 (softmax over classes)
        input = torch.nn.functional.softmax(input, dim=1)

        # 打平，变换形状为 [C, N*D*H*W]
        flattened = flatten(input)

        # 类别平均预测概率统计
        # nominator: (1 - mean_p)
        # denominator: mean_p
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)

        # 计算最终权重
        class_weights = torch.autograd.Variable(nominator / denominator, requires_grad=False)

        return class_weights
#Weighted Cross Entropy Loss (WCE)  动态地为每个类别分配权重，从而减轻类别不平衡对训练的影响
    
class MutualInformation(torch.nn.Module):
    """
    功能：
        计算两张图像 / 分布之间的互信息，用于图像配准、领域自适应、
        distribution alignment 等任务。
        MI 衡量两个随机变量的统计依赖，
        值越大说明两者越相关，其分布越接近。

    实现：
        本实现使用 Gaussian kernel 近似概率分布直方图，
        并在连续域上平滑地估算联合概率与边缘概率。

    参数：
        sigma_ratio : float  
            控制高斯核宽度（决定平滑程度），默认 1。

        minval, maxval : float  
            输入强度值的范围，用于划分直方图的 bin 区间。

        num_bin : int  
            直方图分箱数（bin 数量），决定分辨率。

    返回：
        forward(): 返回负互信息值（MI Loss），用于优化器最小化。
    --------------------------------------------------------------
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        # --- 计算直方图中心点（bins） ---
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = torch.linspace(minval, maxval, num_bin)
        vol_bin_centers = torch.autograd.Variable(vol_bin_centers, requires_grad=False).cuda()

        num_bins = len(bin_centers)

        # --- 计算高斯核标准差 ---
        # 高斯核宽度为相邻两个 bin 的差值均值 × sigma_ratio
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(f"[MutualInformation] Gaussian sigma = {sigma:.6f}")

        # --- 常数项 1 / (2 * sigma**2) ---
        self.preterm = 1.0 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    # --------------------------------------------------------------------------
    def mi(self, y_true, y_pred):
        """
        计算两个输入的互信息值
        ----------------------------------------------------------
        输入:
            y_true: 目标图像/分布, [B, D, H, W] 或 [B, H, W]
            y_pred: 预测图像/分布, 形同 y_true

        输出:
            平均互信息 (Batch 平均)
        ----------------------------------------------------------
        """

        # 限制强度范围
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0., self.max_clip)

        # 展平为二维 [B, voxels]，并增加一个维度用于广播 [B, voxels, 1]
        y_true = y_true.view(y_true.shape[0], -1).unsqueeze(2)
        y_pred = y_pred.view(y_pred.shape[0], -1).unsqueeze(2)

        nb_voxels = y_pred.shape[1]  # 总体素数量

        # --- bin centers reshape 为广播兼容形式 ---
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        # ----------------------------------------------------------------------
        # 计算每个像素强度对各 bin 的高斯响应 (近似概率)
        # soft-assignment：使用 Exp(-||x - bin_center||² / (2σ²))
        # I_a, I_b 对应 y_true 和 y_pred 分布
        # ----------------------------------------------------------------------
        I_a = torch.exp(-self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)  # 归一化

        I_b = torch.exp(-self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # 计算联合分布 P(A,B)
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels

        # 边缘分布 P(A), P(B)
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        # P(A)*P(B)
        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6

        # --- MI = ΣΣ P(A,B) * log(P(A,B) / P(A)P(B)) ---
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)

        # 返回批次平均互信息
        return mi.mean()

    # --------------------------------------------------------------------------
    def forward(self, y_true, y_pred):
        """
        返回负互信息作为损失（便于最小化）
        """
        return -self.mi(y_true, y_pred)
#互信息损失 (Mutual Information Loss)