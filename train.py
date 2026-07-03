import os, glob, datetime,scipy.ndimage
from argparse import ArgumentParser
import numpy as np
import torch.utils.data as Dat
import torch
from Loss import *
from Function import Dataset_OASIS, SpatialTransformer, jacobian_determinant_gpu
from surface_distance import compute_robust_hausdorff, compute_surface_distances
from voxelmorph import VoxelMorph

parser = ArgumentParser()
parser.add_argument("--local_ori", type=float, dest="local_ori", default=0,
                    help="局部方向一致性损失权重（建议范围 1–1000）")
parser.add_argument("--bs_ch", type=int, dest="bs_ch", default=8,
                    help="模型基础通道数")
parser.add_argument("--modelpath", type=str, dest="modelpath", 
                    default='C:\\Users\\19071\\Desktop\\voxelmorph\\Model_weight\\OASIS\\epoch145_dice0.7834.pth',
                    help="模型权重文件路径")
parser.add_argument("--gpu", type=str, dest="gpu", default='0',
                    help="使用的 GPU 编号，如 0 或 0,1")
parser.add_argument("--test_dir", type=str, dest="test_dir", 
                    default='C:/Users/19071/Desktop/neurite-oasis.v1.0/test/',
                    help="测试移动图像文件夹路径")
parser.add_argument("--fixed_dir", type=str, dest="fixed_dir", 
                    default='C:/Users/19071/Desktop/neurite-oasis.v1.0/fixed/',
                    help="固定图像文件夹路径")
parser.add_argument("--classes", type=int, dest="classes", default=36,
                    help="类别数量")



opt = parser.parse_args()  # 解析命令行参数并保存到 opt 对象
bs_ch = opt.bs_ch          # 从命令行获取基础通道数
test_dir = opt.test_dir    # 测试集路径
fixed_dir = opt.fixed_dir  # 固定图像路径
classes = opt.classes      # 类别数量
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu  # 设置可见的 GPU（通过参数控制）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 选用 CUDA 或 CPU
imgshape = (160, 192, 192)  # 输入图像的三维尺寸




'''
test() 是一个医学图像配准模型的验证函数，主要流程是：
加载模型和验证集；
对每对图像进行配准；
计算常见性能指标（Dice、HD95、MSE、NCC、MI、Jacobian）；
打印并记录结果到日志文件中。
'''
def test():
    # 初始化模型与空间变换器（放到 GPU 上）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VoxelMorph(in_channels=2, base_channels=bs_ch).to(device)
    transform = SpatialTransformer().to(device)
    #model = GruopMorph(1, 8, imgshape, groups).cuda()
    #transform = SpatialTransformer().cuda()
    model.eval()
    transform.eval()
    
    # 加载预训练模型权重
    print(f"Loading model from: {opt.modelpath}")
    if not os.path.exists(opt.modelpath):
        raise FileNotFoundError(f"Model file not found: {opt.modelpath}")
    
    checkpoint = torch.load(opt.modelpath, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', None)
        
        # 根据是否有val_loss来决定输出格式
        if val_loss is not None:
            print(f"Model loaded from epoch {epoch}, Val Loss: {val_loss:.4f}")
        else:
            print(f"Model loaded from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Model loaded successfully")

    def forward_register(moving, fixed):
        x_in = torch.cat((moving, fixed), dim=1)
        flows = model(x_in)
        warps = transform(moving, flows)
        return flows, warps

    # 读取测试集（moving 图像）
    test_names = sorted(glob.glob(os.path.join(test_dir, 'OASIS_OAS1_*_MR1')))
    print(f"\nTest images directory: {test_dir}")
    print(f"Total test samples: {len(test_names)}")
    if len(test_names) == 0:
        raise ValueError(f"No test images found in {test_dir}")
    
    valid_generator = Dat.DataLoader(Dataset_OASIS(test_names, norm=False),
                                     batch_size=1, shuffle=False, num_workers=2)
    
    # 读取固定图像
    fixed_names = sorted(glob.glob(os.path.join(fixed_dir, 'OASIS_OAS1_*_MR1')))
    print(f"Fixed image directory: {fixed_dir}")
    if len(fixed_names) == 0:
        raise ValueError(f"No fixed image found in {fixed_dir}")
    print(f"Fixed image: {os.path.basename(fixed_names[0])}")
    
    fixed_generator = Dat.DataLoader(Dataset_OASIS([fixed_names[0]], norm=False),
                                     batch_size=1, shuffle=False, num_workers=1)
    
    # 加载固定图像到内存
    data_fixed = next(iter(fixed_generator))
    Y = data_fixed['image'].to(device)
    Y_label = data_fixed['image_label'].to(device)

    # 初始化指标与设备
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MSES, NCCS, MIS, dice_total_reg, HD95, j_mean, j_std = [], [], [], [], [], [], []
    mi = MutualInformation()

    print("\nTesting...")
    for batch_idx, data in enumerate(valid_generator):
        X, X_label = data['image'].to(device), data['image_label'].to(device)

        with torch.no_grad():  # 关闭梯度计算（测试模式）
            flows, warps = forward_register(X, Y)  # 模型预测形变场与配准结果

            # --- 计算各项指标 ---
            # MSE、NCC、互信息（MI）
            MSES.append(mse(Y, warps).cpu().numpy())
            NCCS.append(-ncc_loss(Y, warps).cpu().numpy())
            MIS.append(-mi(Y, warps).cpu().numpy())

            # 计算配准后的标签并算 Dice
            X_Y_label_5 = transform(X_label, flows, mode='nearest')
            dice_val = dicegup(X_Y_label_5[0, 0], Y_label[0, 0]).cpu().numpy()
            dice_total_reg.append(dice_val)

            # 计算流场 Jacobian 行列式，用于判断形变量的正负分布（光滑性检验）
            f = flows.permute(0, 1, 4, 3, 2)
            j = jacobian_determinant_gpu(f).cpu().numpy()
            j_mean.append(np.mean(j < 0))
            j_std.append(np.std(j))

            # 计算每个标签的 Hausdorff 95 距离
            X_Y_label, Y_label1 = X_Y_label_5[0, 0].cpu().numpy(), Y_label[0, 0].cpu().numpy()
            count, hd95 = 0, 0
            for i in range(1, classes):
                if ((Y_label1 == i).sum() == 0) or ((X_Y_label == i).sum() == 0):
                    continue
                hd95 += compute_robust_hausdorff(
                    compute_surface_distances((Y_label1 == i), (X_Y_label == i), np.ones(3)), 95.)
                count += 1
            if count > 0:
                hd95 /= count
            HD95.append(hd95)

            print(f"step:{len(dice_total_reg)}/{len(test_names)}, reg_dice:{dice_val:.4f}, HD:{hd95:.4f}, "
                  f"HD_mean:{np.mean(HD95):.4f}, MSE:{MSES[-1]:.4f}")

    # 汇总统计结果
    dice_total_reg, j_mean, j_std, HD95 = map(np.array, [dice_total_reg, j_mean, j_std, HD95])
    MSE, NCC, MI = map(np.array, [MSES, NCCS, MIS])
    
    print("\n" + "="*60)
    print("Testing Results Summary:")
    print("="*60)
    print(f"Model path: {opt.modelpath}")
    print(f"Test samples: {len(test_names)}")
    print(f"Fixed image: {os.path.basename(fixed_names[0])}")
    print("-"*60)
    print(f"Registration Dice mean: {dice_total_reg.mean():.4f} (±{dice_total_reg.std():.4f})")
    print(f"Dice median: {np.median(dice_total_reg):.4f}")
    print(f"HD95 mean: {HD95.mean():.4f} (±{HD95.std():.4f})")
    print(f"MSE mean: {MSE.mean():.6f} (±{MSE.std():.6f})")
    #print(f"NCC mean: {NCC.mean():.6f} (±{NCC.std():.6f})")
    print(f"MI mean: {MI.mean():.6f} (±{MI.std():.6f})")
    print(f"Jacobian |J|<0 mean: {j_mean.mean():.6f} (±{j_mean.std():.6f})")
    print(f"Jacobian std mean: {j_std.mean():.6f} (±{j_std.std():.6f})")
    print("="*60)

    # 写入日志文件
    with open("log/test.txt", "a") as log:
        log.write(f"\n{'='*60}\n")
        log.write(f"Test Time: {datetime.datetime.now()}\n")
        log.write(f"Model: {opt.modelpath}\n")
        log.write(f"Test dir: {test_dir}\n")
        log.write(f"Fixed dir: {fixed_dir}\n")
        log.write(f"Samples: {len(dice_total_reg)}\n")
        log.write(f"Reg Dice mean: {dice_total_reg.mean():.4f} (±{dice_total_reg.std():.4f})\n")
        log.write(f"Dice median: {np.median(dice_total_reg):.4f}\n")
        log.write(f"HD95 mean: {HD95.mean():.4f} (±{HD95.std():.4f})\n")
        log.write(f"MSE: {MSE.mean():.6f} (±{MSE.std():.6f})\n")
        log.write(f"NCC: {NCC.mean():.6f} (±{NCC.std():.6f})\n")
        log.write(f"MI: {MI.mean():.6f} (±{MI.std():.6f})\n")
        log.write(f"j_mean: {j_mean.mean():.6f} (±{j_mean.std():.6f})\n")
        log.write(f"j_std: {j_std.mean():.6f} (±{j_std.std():.6f})\n")
        log.write(f"{'='*60}\n")


# ***************计算DICE*********************
def diceval(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1==i).sum()==0) or ((atlas==i).sum()==0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
    if num_count == 0:
        return dice
    else:
        return dice/num_count


# *************计算雅可比行列式***********************
def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    # disp = disp[:,[2,1,0],:]

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (
            jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :,
                                                                                          :]) - \
             jacobian[1, 0, :, :, :] * (
                     jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2,
                                                                                                   1, :, :,
                                                                                                   :]) + \
             jacobian[2, 0, :, :, :] * (
                     jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1,
                                                                                                   1, :, :, :])

    return jacdet



if __name__ == '__main__':
    start = datetime.datetime.now()
    test()
    end = datetime.datetime.now()
    print("Time used:", end - start)
