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

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Initialize LPIPS and SSIM modules
lpips_fn = lpips.LPIPS(net='alex').cuda()  # using AlexNet-based LPIPS
ssim_fn = SSIM(data_range=1.0).cuda()


class CharbonnierLoss(nn.Module):

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x.to('cuda:0') - y.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x.to('cuda:0')), self.laplacian_kernel(y.to('cuda:0')))
        return loss


class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss


def train(train_loader, network, optimizer, scaler):
    losses = AverageMeter()
    char_losses = AverageMeter()
    fft_losses = AverageMeter()
    edge_losses = AverageMeter()

    torch.cuda.empty_cache()
    network.train()
    
    # 使用新的损失函数
    criterion_char = CharbonnierLoss()
    criterion_fft = fftLoss()
    criterion_edge = EdgeLoss()
    
    # 按照指定权重组合损失: loss = char + 0.01*fft + 0.05*edge
    weight_fft = 0.01
    weight_edge = 0.05

    for batch_idx, batch in enumerate(train_loader):
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with autocast(args.no_autocast):
            output = network(source_img)

            # 计算各种损失
            loss_char = criterion_char(output, target_img)
            loss_fft = criterion_fft(output, target_img)
            loss_edge = criterion_edge(output, target_img)

            # 总损失是按权重组合的损失
            loss = loss_char + weight_fft * loss_fft + weight_edge * loss_edge

        losses.update(loss.item())
        char_losses.update(loss_char.item())
        fft_losses.update(loss_fft.item())
        edge_losses.update(loss_edge.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 每100个batch打印一次各个损失值
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}: '
                  f'Total Loss: {losses.val:.4f} ({losses.avg:.4f}), '
                  f'Char Loss: {char_losses.val:.4f} ({char_losses.avg:.4f}), '
                  f'FFT Loss: {fft_losses.val:.4f} ({fft_losses.avg:.4f}), '
                  f'Edge Loss: {edge_losses.val:.4f} ({edge_losses.avg:.4f})')

    # 打印本轮训练的平均损失
    print(f'Epoch Training - '
          f'Average Total Loss: {losses.avg:.4f}, '
          f'Average Char Loss: {char_losses.avg:.4f}, '
          f'Average FFT Loss: {fft_losses.avg:.4f}, '
          f'Average Edge Loss: {edge_losses.avg:.4f}')
    
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
                              batch_size=20,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                              setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=20,
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
