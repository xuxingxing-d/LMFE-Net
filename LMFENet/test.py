import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *


# 设置静态变量
MODEL = 'LMFE-Net'
NUM_WORKERS = 16
DATA_DIR = './data/'
SAVE_DIR = './saved_models/'
RESULT_DIR = './results/'
DATASET = 'Haze1k_moderate'
EXP = 'remote'  # 修改为实际存在的目录名

# 只保留数据集参数作为命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=DATASET, type=str, help='dataset name')
args = parser.parse_args()


def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network, result_dir):
	PSNR = AverageMeter()
	SSIM = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

	for idx, batch in enumerate(test_loader):
		input = batch['source'].cuda()
		target = batch['target'].cuda()

		filename = batch['filename'][0]

		with torch.no_grad():
			output = network(input).clamp_(-1, 1)

			# [-1, 1] to [0, 1]
			output = output * 0.5 + 0.5
			target = target * 0.5 + 0.5

			psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

			_, _, H, W = output.size()
			down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
			ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))), 
							F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))), 
							data_range=1, size_average=False).item()				

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)

		print('Test: [{0}]\t'
			  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
			  'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
			  .format(idx, psnr=PSNR, ssim=SSIM))

		f_result.write('%s,%.02f,%.03f\n'%(filename, psnr_val, ssim_val))

		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		write_img(os.path.join(result_dir, 'imgs', filename), out_img)

	f_result.close()

	os.rename(os.path.join(result_dir, 'results.csv'), 
			  os.path.join(result_dir, '%.02f | %.04f.csv'%(PSNR.avg, SSIM.avg)))


if __name__ == '__main__':
	# 使用静态变量替代原来的 args.xxx
	if MODEL == 'LMFE-Net':
		network = lmfe_net()
	else:
		network = eval(MODEL.replace('-', '_'))()
	network.cuda()
	saved_model_dir = os.path.join(SAVE_DIR, EXP, MODEL+'.pth')

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + MODEL)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	# 使用静态变量替代原来的 args.xxx
	dataset_dir = os.path.join(DATA_DIR, args.dataset)  # 保留数据集参数
	test_dataset = PairLoader(dataset_dir, 'test', 'test')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=NUM_WORKERS,
							 pin_memory=True)

	# 使用静态变量替代原来的 args.xxx
	result_dir = os.path.join(RESULT_DIR, args.dataset, MODEL)
	test(test_loader, network, result_dir)