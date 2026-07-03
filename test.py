import os, glob, sys, math
from argparse import ArgumentParser
import torch.utils.data as Data
from Loss import *
from Function import Dataset_OASIS, SpatialTransformer
import numpy as np
import torch
from voxelmorph import VoxelMorph

parser = ArgumentParser()
parser.add_argument("--lr", type=float, dest="lr", default=1e-4, help="学习率")
parser.add_argument("--max_epoch", type=int, dest="max_epoch", default=500, help="最大训练epoch数")
parser.add_argument("--smooth", type=float, dest="smooth", default=0.5, help="形变平滑损失权重")
parser.add_argument("--sim", type=float, dest="sim", default=1.0, help="相似性损失权重")
parser.add_argument("--checkpoint", type=int, dest="checkpoint", default=1, help="保存模型间隔（epoch数）")
parser.add_argument("--batch_size", type=int, dest="batch_size", default=1, help="训练批次大小")
parser.add_argument("--bs_ch", type=int, dest="bs_ch", default=8, help="网络基础通道数")
parser.add_argument("--modelname", type=str, dest="model_name", default='', help="保存模型名称")
parser.add_argument("--gpu", type=str, dest="gpu", default='0', help="使用的GPU ID")
parser.add_argument("--train_dir", type=str, dest="train_dir", default='C:/Users/19071/Desktop/neurite-oasis.v1.0/train/', help="训练移动图像文件夹路径")
parser.add_argument("--fixed_dir", type=str, dest="fixed_dir", default='C:/Users/19071/Desktop/neurite-oasis.v1.0/fixed/', help="固定图像文件夹路径")
parser.add_argument("--val_dir", type=str, dest="val_dir", default='C:/Users/19071/Desktop/neurite-oasis.v1.0/val/', help="验证移动图像文件夹路径")

opt = parser.parse_args()
lr = opt.lr
bs_ch = opt.bs_ch
batch_size = opt.batch_size
n_checkpoint = opt.checkpoint
# 损失权重参数
sim_weight = opt.sim
smooth_weight = opt.smooth
model_name = opt.model_name
max_epoch = opt.max_epoch
train_dir = opt.train_dir
fixed_dir = opt.fixed_dir
val_dir = opt.val_dir
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
imgshape = (160, 192, 192)



def train():
    # 初始化模型与相关组件
    model = VoxelMorph(in_channels=2, base_channels=bs_ch).cuda()
    loss_similarity = NCC(win=9).cuda()
    transfor = SpatialTransformer().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)

    def forward_register(moving, fixed):
        x_in = torch.cat((moving, fixed), dim=1)
        flows = model(x_in)
        warps = transfor(moving, flows)
        smo = smoothloss(flows)
        return flows, warps, smo

    # 模型文件保存路径
    model_dir = 'C:/Users/19071/Desktop/ViT-V-Net_for_3D_Image_Registration_Pytorch-main/Model_weight/yuan_group'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    epoch_start = 0
    load_model = False   # 设置为 True 以载入已有模型
    best_dice = 0.0  # 记录最佳Dice（越大越好）

    # 载入已有模型（若启用）
    if load_model is True:
        model_path = 'C:/Users/19071/Desktop/GroupMorp/Model_weight/yuan_group/epoch0_loss1.0000.pth'
        epoch_start = 1 # 根据实际训练进度设置
        # 加载checkpoint(包含numpy数据的旧格式模型需要weights_only=False)
        checkpoint = torch.load(model_path, weights_only=False)
        # 判断是否是新格式(字典)还是旧格式(直接state_dict)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'best_dice' in checkpoint:
                best_dice = checkpoint['best_dice']
                print(f"Loaded best_dice: {best_dice:.4f}")
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from epoch {epoch_start}")

    # 加载训练移动图像
    train_names = sorted(glob.glob(os.path.join(train_dir, 'OASIS_OAS1_*_MR1')))
    training_generator = Data.DataLoader(Dataset_OASIS(train_names, norm=False),
                                         batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 加载固定图像（只有一张）
    fixed_names = sorted(glob.glob(os.path.join(fixed_dir, 'OASIS_OAS1_*_MR1')))
    if len(fixed_names) == 0:
        raise ValueError(f"No fixed image found in {fixed_dir}")
    fixed_generator = Data.DataLoader(Dataset_OASIS([fixed_names[0]], norm=False),
                                      batch_size=batch_size, shuffle=False, num_workers=2)

    # 加载固定图像到内存
    data_fixed = next(iter(fixed_generator))
    fixed_image = data_fixed['image'].cuda().float()
    fixed_label = data_fixed['image_label'].cuda().float()
    print(f"Pairs per epoch: {len(train_names)}")

    # 加载验证集
    val_names = sorted(glob.glob(os.path.join(val_dir, 'OASIS_OAS1_*_MR1')))
    valid_generator = Data.DataLoader(Dataset_OASIS(val_names, norm=False),
                                      batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Validation pairs: {len(val_names)}")
    print(f"Using batch size: {batch_size}")

    # 主训练循环 - 基于epoch
    for epoch in range(epoch_start, max_epoch):
        print(f'\n=== Epoch {epoch}/{max_epoch-1} ===')
        
        
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        epoch_sim_loss = 0.0
        epoch_smo_loss = 0.0
        total_pairs = 0

        for _, data in enumerate(training_generator):
            X = data['image'].cuda().float()
            # 使用固定的单张固定图像
            Y = fixed_image

            # 前向传播
            flows, warps, smo = forward_register(X, Y)

            # 组合总损失（仅使用相似性损失和平滑损失）
            sim = loss_similarity(warps, Y)
            # 计算乘完权重后的各部分损失
            sim_weighted = sim_weight * sim
            smo_weighted = smooth_weight * smo
            loss = sim_weighted + smo_weighted

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 累积损失（使用乘完权重后的值）
            epoch_loss += loss.item()
            epoch_sim_loss += sim_weighted.item()
            epoch_smo_loss += smo_weighted.item()
            total_pairs += 1

            # 每10对打印一次
            if total_pairs % 10 == 0:
                sys.stdout.write(
                    "\r" + 'Epoch {0} - Pair [{1}/{2}] - loss: {3:.4f} - sim: {4:.4f} - smo: {5:.4f} - lr: {6:.2e}'.format(
                        epoch, total_pairs, len(train_names),
                        loss.item(), sim_weighted.item(), smo_weighted.item(), lr))
                sys.stdout.flush()

        # Epoch结束，计算平均损失
        avg_loss = epoch_loss / total_pairs
        avg_sim = epoch_sim_loss / total_pairs
        avg_smo = epoch_smo_loss / total_pairs
        
        print(f'\nEpoch {epoch} Training - Avg Loss: {avg_loss:.4f}, Sim: {avg_sim:.4f}, Smo: {avg_smo:.4f}')
        
        # 写入日志
        log_dir = "yuan_log/loss.txt"
        with open(log_dir, "a") as log:
            log.write(f'Epoch {epoch} - Avg Loss: {avg_loss:.4f} - Sim: {avg_sim:.4f} - Smo: {avg_smo:.4f} - LR: {lr:.2e}\n')

        # 每隔 n_checkpoint 个epoch进行验证和保存
        if (epoch % n_checkpoint == 0) or (epoch == max_epoch - 1):
            # 验证
            dice_total = []
            print("\nValidating...")

            model.eval()
            with torch.no_grad():
                for batch_idx_val, data_val in enumerate(valid_generator):
                    X_val = data_val['image'].cuda().float()
                    X_label_val = data_val['image_label'].cuda().float()

                    # 使用固定的单张固定图像进行验证
                    Y_val = fixed_image
                    Y_label_val = fixed_label

                    flows_val, _, _ = forward_register(X_val, Y_val)

                    # 计算验证Dice（使用最近邻插值保证标签一致）
                    X_to_Y_label_val = transfor(X_label_val, flows_val, mode='nearest')[0, 0, :, :, :]
                    dice_val = dicegup(X_to_Y_label_val, Y_label_val[0, 0, :, :, :])
                    dice_total.append(dice_val.cpu().numpy())

                    # 打印验证进度
                    sys.stdout.write(f"\rValidation progress: {batch_idx_val + 1}/{len(valid_generator)}")
                    sys.stdout.flush()

            model.train()

            # 计算验证结果
            if len(dice_total) > 0:
                dice_total = np.array(dice_total)
                dice_mean = dice_total.mean()
                print(f"\nValidation Dice mean: {dice_mean:.4f} (Total pairs: {len(dice_total)})")

                if dice_mean > best_dice:
                    best_dice = dice_mean
                    print(f'New best Dice: {best_dice:.4f}')
            else:
                dice_mean = 0.0
                print("No validation performed (no validation data)")

            # 使用dice值和epoch命名保存模型
            modelname = model_dir + '/' + model_name + f'epoch{epoch}_dice{dice_mean:.4f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': dice_mean,
                'best_dice': best_dice,
            }, modelname)
            print(f'Model saved: {modelname}')

            # 记录验证结果
            log_dir = "yuan_log/val.txt"
            with open(log_dir, "a") as log:
                log.write(f"Epoch {epoch} - Dice mean: {dice_mean:.4f} - Total pairs: {len(dice_total)} - Best Dice: {best_dice:.4f}\n")

    print(f'\nTraining completed! Best Dice: {best_dice:.4f}')

if __name__ == '__main__':
    train()
