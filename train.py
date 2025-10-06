import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import lpips  # for LPIPS
from thop import profile  # 导入 thop 用于计算 FLOPs 和参数
from torch.fft import fft2, ifft2
from utils import AverageMeter
from datasets.loader import PairLoader
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='LMFE—Net', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--dataset', default='Haze1k_moderate', type=str, help='dataset name')
parser.add_argument('--exp', default='remote', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')  # 这里修改为单个 GPU
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Initialize LPIPS and SSIM modules
lpips_fn = lpips.LPIPS(net='alex').cuda()  # using AlexNet-based LPIPS
ssim_fn = SSIM(data_range=1.0).cuda()


class SpatialLoss(nn.Module):
    def __init__(self):
        super(SpatialLoss, self).__init__()

    def forward(self, x, y):
        return torch.mean(torch.abs(x - y))

class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def forward(self, x, y):
        # Compute the 2D FFT of both images
        x_fft = fft2(x, dim=(-2, -1))
        y_fft = fft2(y, dim=(-2, -1))

        # Separate amplitude and phase components
        x_amp = torch.abs(x_fft)
        y_amp = torch.abs(y_fft)
        x_phase = torch.angle(x_fft)
        y_phase = torch.angle(y_fft)

        # Compute the L1 loss for amplitude and phase
        amp_loss = torch.mean(torch.abs(x_amp - y_amp))
        phase_loss = torch.mean(torch.abs(x_phase - y_phase))

        return amp_loss + phase_loss

def train(train_loader, network, optimizer, scaler):
    losses = AverageMeter()

    torch.cuda.empty_cache()
    network.train()
    criterion_spa = SpatialLoss()
    criterion_fre = FrequencyLoss()
    alpha = 0.05  # Weight for frequency loss

    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with autocast(args.no_autocast):
            output = network(source_img)

            # Calculate spatial loss
            loss_spa = criterion_spa(output, target_img)

            # Calculate frequency loss
            loss_fre = criterion_fre(output, target_img)

            # Total loss is the sum of spatial loss and weighted frequency loss
            loss = loss_spa + alpha * loss_fre

        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg

def valid(val_loader, network):
    PSNR = AverageMeter()
    SSIM_values = AverageMeter()
    LPIPS_values = AverageMeter()

    torch.cuda.empty_cache()
    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():
            output = network(source_img).clamp_(-1, 1)

        # PSNR calculation
        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

        # SSIM calculation
        ssim_val = ssim_fn(output * 0.5 + 0.5, target_img * 0.5 + 0.5)
        SSIM_values.update(ssim_val.item(), source_img.size(0))

        # LPIPS calculation
        lpips_val = lpips_fn(output * 0.5 + 0.5, target_img * 0.5 + 0.5)
        LPIPS_values.update(lpips_val.mean().item(), source_img.size(0))

    print(f"Current PSNR: {PSNR.avg:.4f}")
    return PSNR.avg, SSIM_values.avg, LPIPS_values.avg

def plot_curves(train_losses, psnr_values, ssim_values, lpips_values, save_dir):
    epochs = range(1, len(train_losses) + 1)

    # Plot loss curve
    plt.figure()
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))

    # Plot PSNR curve
    plt.figure()
    plt.plot(epochs, psnr_values, 'r', label='PSNR')
    plt.title('Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'psnr_curve.png'))

    # Plot SSIM curve
    plt.figure()
    plt.plot(epochs, ssim_values, 'g', label='SSIM')
    plt.title('Validation SSIM')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'ssim_curve.png'))

    # Plot LPIPS curve
    plt.figure()
    plt.plot(epochs, lpips_values, 'm', label='LPIPS')
    plt.title('Validation LPIPS')
    plt.xlabel('Epochs')
    plt.ylabel('LPIPS')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'lpips_curve.png'))

    plt.show()

if __name__ == '__main__':
    # === Step 1: 创建原始模型（不带 DataParallel）===
    model = eval(args.model.replace('-', '_'))().cuda()

    # === Step 2: 使用原始模型计算 FLOPs 和参数量 ===
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    macs, params = profile(model, inputs=(input_tensor,))
    print(f"Total MACs: {macs / 1e9:.2f} G")  # MACs 是 FLOPs 的近似
    print(f"Total Params: {params / 1e6:.2f} M")

    # === Step 3: 将该模型包装为 DataParallel 并用于训练 ===
    network = nn.DataParallel(model).cuda()  # 包装为 DataParallel 用于训练

    # 加载配置文件
    setting_filename = os.path.join('configs', args.exp, args.model+'.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                                setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=9,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                              setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=9,
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    # Lists to store loss, PSNR, SSIM, and LPIPS for plotting
    train_losses = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
        print('==> Start training, current model name: ' + args.model)

        best_psnr = 0  # 初始化 best_psnr
        for epoch in tqdm(range(setting['epochs'] + 1)):
            loss = train(train_loader, network, optimizer, scaler)
            train_losses.append(loss)

            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr, avg_ssim, avg_lpips = valid(val_loader, network)
                psnr_values.append(avg_psnr)
                ssim_values.append(avg_ssim)
                lpips_values.append(avg_lpips)

                # 更新并打印 best_psnr
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model+'.pth'))

                print(f"Best PSNR: {best_psnr:.4f}")  # 打印当前 best_psnr

        # 绘制损失、PSNR、SSIM和LPIPS曲线并保存为 PNG
        plot_curves(train_losses, psnr_values, ssim_values, lpips_values, save_dir)

    else:
        print('==> Existing trained model')
        exit(1)