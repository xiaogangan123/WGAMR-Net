import nibabel as nib
from torch.utils import data
import itertools
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F

def upsample(img, is_flow, scale_factor=2.0, align_corners=True):
    """
    参数:
        img: [B, C, D, H, W] 张量
        is_flow: 是否为形变场，若为 True 则插值后按比例缩放数值
        scale_factor: 缩放倍数
        align_corners: 插值方式参数
    返回:
        上采样后的张量
    """
    img_resized = nn.functional.interpolate(
        img,
        scale_factor=scale_factor,
        mode='trilinear',
        align_corners=align_corners
    )
    if is_flow:
        img_resized *= scale_factor
    return img_resized
#上采样 3D 图像或形变场

class VecInt(nn.Module):
    def __init__(self, nsteps):
        super().__init__()
        assert nsteps >= 0, f"nsteps should be >= 0, found: {nsteps}"
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** nsteps)  # 缩放因子
        self.transformer = SpatialTransformer()  # 应用形变
    def forward(self, vec):
        # 缩放初始速度场
        vec = vec * self.scale
        # 迭代积分 nsteps 次
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec
#速度场积分模块（Scaling and Squaring）用于将速度场(velocity field)转换为位移场(displacement field)，
    

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, src, flow, mode='bilinear'):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        grid = grid.cuda()
        # grid = grid

        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=mode)
#空间变换器模块，用于根据给定的形变场(flow field)对输入图像(src)进行空间变换。

def save_img(I_img, savename):
    """
    参数:
        I_img: 要保存的图像数组 (numpy ndarray)
        savename: 保存路径（包含文件名）
    """
    affine = np.diag([1, 1, 1, 1])  # 单位仿射矩阵
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)
#保存 3D 图像为 NIfTI 格式文件 (.nii)


def save_flow(I_img, savename):
    """
    参数:
        I_img: 形变场数组 (numpy ndarray)，通常形状为 [X, Y, Z, 3]
        savename: 保存路径（包含文件名）
    """
    affine = np.diag([1, 1, 1, 1])  # 单位仿射矩阵
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)
#保存 3D 形变场为 NIfTI 格式文件 (.nii)

class Dataset_OASIS(Data.Dataset):# OASIS 数据集接口，用于加载医学图像配准任务所需的样本。
  'Characterizes a dataset for PyTorch'
  def __init__(self, names, norm=False):
    """
    参数:
        names: 图像文件名或路径列表
        norm: 是否对图像进行归一化
    """
    super(Dataset_OASIS, self).__init__()
    self.norm = norm              # 是否归一化标志
    self.index_pair = list(names) # 样本索引或路径列表
  #初始化 OASIS 数据集
  def __len__(self):
    return len(self.index_pair)
  #返回数据集中样本的总数
  def __getitem__(self, index):
    """
    参数:
        index: 数据样本索引
    返回:
        output: dict, 包含
            - 'image': 归一化后的图像张量
            - 'image_label': 对应的分割标签张量
            - 'index': 当前样本索引
    """
    # 构造图像与标签的路径
    moved_img0 = self.index_pair[index] + "/aligned_norm.nii.gz"
    moved_label0 = self.index_pair[index] + "/aligned_seg35.nii.gz"

    # 读取并裁剪 NIfTI 图像与标签到指定尺寸
    moved_img = load_4D_with_crop(moved_img0, cropx=160, cropy=192, cropz=192)
    moved_label = load_4D_with_crop(moved_label0, cropx=160, cropy=192, cropz=192)
    # 图像归一化（如果启用）
    if self.norm:
        moved_img = imgnorm(moved_img)
    # 转换为 PyTorch 张量
    moved_img = torch.from_numpy(moved_img)
    moved_label = torch.from_numpy(moved_label)
    # 打包输出字典
    output = {
        'image': moved_img.float(),
        'image_label': moved_label.float(),
        'index': index
    }
    return output
  #根据索引加载并返回一个样本（图像与对应标签）


def load_4D(name):
    """
    参数:
        name: str, 图像文件路径 (.nii / .nii.gz)

    返回:
        X2: ndarray, 形状为 (1, D, H, W)
    """
    X0 = nib.load(name)             # 读取 NIfTI 图像对象
    X1 = X0.get_fdata()             # 获取图像数据为 numpy 数组
    X2 = np.reshape(X1, (1,) + X1.shape)  # 增加通道维 (C=1)
    return X2
#读取 NIfTI 图像并在前面增加一维（通常作为通道维）

def load_4D_with_crop(name, cropx, cropy, cropz):
    """
    参数:
        name:  str, 图像文件路径 (.nii / .nii.gz)
        cropx: int, 在 x 方向上的裁剪尺寸
        cropy: int, 在 y 方向上的裁剪尺寸
        cropz: int, 在 z 方向上的裁剪尺寸

    返回:
        X: ndarray, 形状为 (1, cropx, cropy, cropz)
    """
    # 读取 NIfTI 文件并提取为 numpy 数组
    X = nib.load(name).get_fdata()
    # 原始体数据尺寸
    x, y, z = X.shape
    # 计算中心裁剪的起始坐标
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    startz = z // 2 - cropz // 2
    # 在三个维度上进行中心裁剪
    X = X[startx:startx+cropx, starty:starty+cropy, startz:startz+cropz]
    # 增加通道维度 (C=1)，以符合网络输入格式
    X = np.reshape(X, (1,) + X.shape)
    return X
# 读取 NIfTI 图像文件，并在中心位置裁剪成指定大小的 3D patch，同时在最前面增加一个通道维度。

def imgnorm(img):
    """
    参数:
        img: ndarray，原始图像数据
    返回:
        norm_img: ndarray，归一化后的图像
    """
    max_v = np.max(img)
    min_v = np.min(img)
    # 避免除零错误（图像所有像素相等时）
    if max_v == min_v:
        return np.zeros_like(img)
    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img
#将输入图像进行归一化（Min-Max Scaling）到 [0, 1] 区间

def generate_grid_unit(imgshape):
    """
    参数:
        imgshape: tuple (D, H, W)  图像的三维尺寸
    返回:
        grid: ndarray, 形状为 (D, H, W, 3)
              最后一个维度分别存储 (x, y, z) 坐标
    """
    # 每个维度生成归一化坐标，中心点为 0，范围 [-1, 1]
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    # 生成 3D 网格坐标
    mesh = np.meshgrid(z, y, x)  # 注意 meshgrid 输出顺序 (z, y, x) 
                                 # shape: [3, H, W, D] 后续要调整轴
    # 调整轴顺序到 (D, H, W, 3)
    grid = np.rollaxis(np.array(mesh), 0, 4)  # 变成 (H, W, D, 3)
    grid = np.swapaxes(grid, 0, 2)            # 变成 (D, W, H, 3)
    grid = np.swapaxes(grid, 1, 2)            # 变成 (D, H, W, 3)
    return grid
#生成一个 [-1, 1] 范围的三维归一化坐标网格 (z, y, x)，用于体数据的空间变形或插值。


def jacobian_determinant_gpu(dense_flow):
    """
    参数:
        dense_flow: torch.Tensor, 形状为 [B, 3, H, W, D]
            三维位移场张量，每个体素存储 (dx, dy, dz)。
    返回:
        jac_det: torch.Tensor, 形状约为 [H-4, W-4, D-4]
            对应每个体素位置的 Jacobian determinant 值。
            值 > 0 表示局部体积保持或扩张；
            值 < 0 表示局部空间折叠 (folding)。
    功能说明:
        本函数利用 3D 卷积核近似计算三维位移场在 z, y, x 三个方向上的梯度，
        进而构造出每个体素的 Jacobian 矩阵 J = I + ∇u，
        并根据行列式公式计算 det(J)，用于衡量局部体积变化。
    """
    # 提取三维形变场的尺寸
    _, _, H, W, D = dense_flow.shape

    # 调整维度顺序为 [z, y, x]，方便后续计算梯度
    dense_pix = dense_flow[:, [2, 1, 0], :].to(dense_flow.device)

    # ----- 构造三个方向的卷积核（中心差分）-----
    # z方向梯度卷积核
    gradz = nn.Conv3d(3, 3, (3, 1, 1), padding=(1, 0, 0), bias=False, groups=3)
    gradz.weight.data[:, 0, :, 0, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradz.to(dense_flow.device)

    # y方向梯度卷积核
    grady = nn.Conv3d(3, 3, (1, 3, 1), padding=(0, 1, 0), bias=False, groups=3)
    grady.weight.data[:, 0, 0, :, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    grady.to(dense_flow.device)

    # x方向梯度卷积核
    gradx = nn.Conv3d(3, 3, (1, 1, 3), padding=(0, 0, 1), bias=False, groups=3)
    gradx.weight.data[:, 0, 0, 0, :] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradx.to(dense_flow.device)

    # ----- 计算雅可比矩阵的每个分量 -----
    with torch.no_grad():
        # 拼接三个方向上的梯度，得到 ∇u
        # 并加上单位矩阵 I，形成每个点的 Jacobian 矩阵 J = I + ∇u
        jacobian = torch.cat((gradz(dense_pix), grady(dense_pix), gradx(dense_pix)), 0) \
                   + torch.eye(3, 3).view(3, 3, 1, 1, 1).to(dense_flow.device)

        # 裁剪边界防止卷积造成的尺寸偏差
        jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]

        # 计算 3x3 Jacobian 的行列式（体素级）
        jac_det = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :]
                - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) \
                  - jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :]
                - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) \
                  + jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :]
                - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
    # 返回雅可比行列式张量
    return jac_det
#计算三维形变场 (dense_flow) 的雅可比行列式 (Jacobian determinant)。

import re
def process_label():
    """
    功能说明:
        FreeSurfer 的分割结果中，每个结构（例如皮层区、皮层下结构）都有一个整数编号。
        此函数读取标签定义文件 (seg35_labels.txt)，结合给出的标签表 seg_table，
        生成一个索引到标签名称的映射字典，用于后续处理或可视化。
    返回:
        dict: {索引号 -> 标签名} 的字典，例如 {0: 'Unknown', 1: 'Left-Cerebral-White-Matter', ...}
    """
    # 对应的 FreeSurfer 分割标签编号列表（筛选所需标签）
    #process labeling information for FreeSurfer
    # seg_table = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
    #                       28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
    #                       63, 72, 77, 80, 85, 251, 252, 253, 254, 255]
    seg_table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                 31, 32, 33, 34, 35]
    # 打开标签定义文件（每一行的格式形如 "编号 名称  ...其它信息"）
    file1 = open('seg35_labels.txt', 'r')
    Lines = file1.readlines()
    dict = {}         # 存储 seg_i -> 标签名 的映射
    seg_i = 0         # 自增的标签索引
    seg_look_up = []  # 可选存储更多信息：[索引, 原编号, 名称]
    # 遍历指定的标签编号列表
    for seg_label in seg_table:
        for line in Lines:
            # 用正则去掉多余空格，然后按空格分割
            line = re.sub(' +', ' ', line).split(' ')
            # 如果此行不是有效的数字开头则跳过
            try:
                int(line[0])
            except:
                continue
            # 如果行的编号匹配当前目标 seg_label，则记录信息
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])  # 记录索引、编号、名称
                dict[seg_i] = line[1]  # 记录字典映射：索引号 → 标签名
        seg_i += 1
    return dict
#读取并处理 FreeSurfer 的标签配置信息，构建一个用于查找标签名的字典。

def csv_writter(line, name):
    """
    参数:
        line : str
            要写入文件的一行内容（通常是逗号分隔的字符串，如 "value1,value2,value3"）。
        name : str
            文件名（不含扩展名）。函数会自动创建或在已有文件中追加写入到 <name>.csv。
    功能说明:
        - 以追加模式 ('a') 打开一个名为 <name>.csv 的文件；
        - 将传入的字符串 line 写入文件并换行；
        - 如果文件不存在则自动创建；
        - 该函数不使用 Python 的 csv 库，但适用于简单文本追加的场景。
    """
    # 使用上下文管理器安全地打开文件，确保写入后自动关闭
    with open(name + '.csv', 'a') as file:
        file.write(line)   # 写入一行内容
        file.write('\n')   # 手动添加换行符
#将一行字符串追加写入到 CSV 文件中

def dice_val_substruct(y_pred, y_true, std_idx):
    """
    参数:
        y_pred : torch.Tensor
            模型预测的分割标签，形状通常为 [B, 1, H, W, D] 或 [B, H, W, D]，
            每个体素存储类别编号（整数）。
        y_true : torch.Tensor
            Ground Truth（真实）分割标签，形状与 y_pred 相同。
        std_idx : int 或 str
            样本或标准的标识，用于生成行数据的第一列 (如 "p_3")。
    返回:
        line : str
            形如 'p_<std_idx>,dsc_class0,dsc_class1,...,dsc_class35' 的字符串，
            数值是针对每个类别（共36类）的 Dice 系数。
    """
    # 在不计算梯度的情况下进行 one-hot 编码
    with torch.no_grad():
        # 预测结果进行 one-hot 编码，num_classes=36 (假设标签类别数为36)
        y_pred = nn.functional.one_hot(y_pred, num_classes=36)
        y_pred = torch.squeeze(y_pred, 1)  # 去掉可能存在的多余的通道维度
        # 调整维度顺序为 [B, C, H, W, D]，使类别维在第二维度
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()

        # 真实标签同样进行 one-hot 编码和维度调整
        y_true = nn.functional.one_hot(y_true, num_classes=36)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()

    # 转为 NumPy 数组以便后续计算
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    # 初始化输出字符串，第一列是样本标识
    line = 'p_{}'.format(std_idx)

    # 遍历每一个类别（0~35）
    for i in range(36):
        # 获取当前类别的预测二值图和真实二值图
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]

        # 计算交集体素数（intersection）
        intersection = pred_clus * true_clus
        intersection = intersection.sum()

        # 计算预测与真实的总体素数（union）
        union = pred_clus.sum() + true_clus.sum()

        # 计算 Dice 系数： 2 * (交集) / (总和)
        dsc = (2. * intersection) / (union + 1e-5)  # 加 1e-5 防止除零

        # 将当前类别的 Dice 值追加到 CSV 字符串
        line = line + ',' + str(dsc)

    return line
#计算分割结果的 Dice 系数（逐类别），并返回一行 CSV 格式的字符串。


# IXI 数据集类
class Dataset_IXI(Data.Dataset):
    """IXI 数据集接口，用于加载存储为 .pkl 格式的医学图像配准任务数据"""
    def __init__(self, file_list, atlas_file, norm=True):
        """
        参数:
            file_list: pkl 文件路径列表 (moving images)
            atlas_file: atlas图像的pkl文件路径 (fixed image)
            norm: 是否对图像进行归一化
        """
        super(Dataset_IXI, self).__init__()
        self.file_list = file_list
        self.atlas_file = atlas_file
        self.norm = norm
        
        # 加载atlas图像（固定图像）
        import pickle
        with open(atlas_file, 'rb') as f:
            self.atlas_data = pickle.load(f)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        """
        参数:
            index: 数据样本索引
        返回:
            output: dict, 包含
                - 'image': moving image (待配准的图像)
                - 'label': fixed image (参考图像/atlas)
        """
        import pickle
        
        # 加载moving image
        with open(self.file_list[index], 'rb') as f:
            moving_data = pickle.load(f)
        
        # 如果数据已经有4维 (C, D, H, W)，只取第一个通道
        if len(moving_data.shape) == 4:
            moving_data = moving_data[0]  # 取第一个通道
        
        # 如果atlas数据已经有4维，也只取第一个通道
        atlas_data = self.atlas_data
        if len(atlas_data.shape) == 4:
            atlas_data = atlas_data[0]
        
        # 归一化
        if self.norm:
            moving_data = imgnorm(moving_data)
            atlas_data = imgnorm(atlas_data)
        
        # 确保数据格式正确 (C, D, H, W) - 添加通道维度
        if len(moving_data.shape) == 3:
            moving_data = np.expand_dims(moving_data, axis=0)
        if len(atlas_data.shape) == 3:
            atlas_data = np.expand_dims(atlas_data, axis=0)
        
        # 转换为 PyTorch 张量
        moving_img = torch.from_numpy(moving_data)
        fixed_img = torch.from_numpy(atlas_data)
        
        output = {
            'image': moving_img.float(),
            'label': fixed_img.float(),
        }
        return output



