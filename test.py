import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
import lpips  # 导入lpips库

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-b', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
args = parser.parse_args()

num = 0
PSNRsum = 0.0
SSIMsum = 0.0
LPIPSsum = 0.0  # 新增LPIPS总和变量

# LPIPS计算器
loss_fn_lpips = lpips.LPIPS(net='alex').cuda()

def single(save_dir):
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        # 过滤掉包含 "total_ops" 和 "total_params" 的键
        if "total_ops" in k or "total_params" in k:
            continue
        name = k[7:] if k.startswith("module.") else k  # 移除 'module.' 前缀
        new_state_dict[name] = v

    return new_state_dict


def test(test_loader, network, result_dir):
    global num, PSNRsum, SSIMsum, LPIPSsum
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    LPIPS_meter = AverageMeter()  # 用于存储LPIPS的平均值
    torch.cuda.empty_cache()

    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        num += 1
        input = batch['source'].cuda()
        target = batch['target'].cuda()

        filename = batch['filename'][0]

        with torch.no_grad():
            output = network(input).clamp_(-1, 1)

            # [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            # 计算 PSNR
            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            # 计算 SSIM
            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(
                F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                data_range=1, size_average=False).item()

            # 计算 LPIPS
            lpips_val = loss_fn_lpips(output, target).item()

            PSNR.update(psnr_val)
            SSIM.update(ssim_val)
            LPIPS_meter.update(lpips_val)  # 更新LPIPS
            print('Test: [{0}]\t'
                  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
                  'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})\t'
                  'LPIPS: {lpips_val:.03f} ({lpips_avg:.03f})'
                  .format(idx, psnr=PSNR, ssim=SSIM, lpips_val=lpips_val, lpips_avg=LPIPS_meter.avg))

            PSNRsum += psnr_val
            SSIMsum += ssim_val
            LPIPSsum += lpips_val  # 记录LPIPS总和
            f_result.write('%s,%.02f,%.03f,%.03f\n' % (filename, psnr_val, ssim_val, lpips_val))

            out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
            write_img(os.path.join(result_dir, 'imgs', filename), out_img)

    PSNRaver = PSNRsum / num
    SSIMaver = SSIMsum / num
    LPIPSaver = LPIPSsum / num  # 计算LPIPS平均值
    print('PSNR平均值为{:.02f}'.format(PSNRaver))
    print('SSIM平均值为{:.03f}'.format(SSIMaver))
    print('LPIPS平均值为{:.03f}'.format(LPIPSaver))  # 打印LPIPS平均值
    f_result.close()

    os.rename(os.path.join(result_dir, 'results.csv'),
              os.path.join(result_dir, '{:.02f} | {:.04f} | {:.03f}.csv'.format(PSNRaver, SSIMaver, LPIPSaver)))


if __name__ == '__main__':
    network = eval(args.model.replace('-', '_'))()
    network.cuda()
    saved_model_dir = os.path.join(args.save_dir, args.exp, args.model + '.pth')

    if os.path.exists(saved_model_dir):
        print('==> Start testing, current model name: ' + args.model)
        network.load_state_dict(single(saved_model_dir))
    else:
        print('==> No existing trained model!')
        exit(0)

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    test_dataset = PairLoader(dataset_dir, 'test', 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    result_dir = os.path.join(args.result_dir, args.dataset, args.model)
    test(test_loader, network, result_dir)
