import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter
from datasets.loader import PairLoader
from models import *


# 设置静态变量
MODEL = 'LMFE-Net'
NUM_WORKERS = 16
NO_AUTOCAST = True
SAVE_DIR = './saved_models/'
DATA_DIR = './data/'
LOG_DIR = './logs/'
DATASET = 'Haze1k_moderate'
EXP = 'remote'  # 修改为实际存在的目录名
GPU = '0,1,2,3'

# 只保留数据集参数作为命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=DATASET, type=str, help='dataset name')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = GPU


def train(train_loader, network, criterion, optimizer, scaler):
	losses = AverageMeter()

	torch.cuda.empty_cache()
	
	network.train()

	for batch in train_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with autocast(NO_AUTOCAST):
			output = network(source_img)
			loss = criterion(output, target_img)

		losses.update(loss.item())

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

	return losses.avg


def valid(val_loader, network):
	PSNR = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	for batch in val_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with torch.no_grad():							# torch.no_grad() may cause warning
			output = network(source_img).clamp_(-1, 1)		

		mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		PSNR.update(psnr.item(), source_img.size(0))

	return PSNR.avg


if __name__ == '__main__':
	# 使用静态变量替代原来的 args.xxx
	setting_filename = os.path.join('configs', EXP, MODEL+'.json')
	if not os.path.exists(setting_filename):
		setting_filename = os.path.join('configs', EXP, 'default.json')
	with open(setting_filename, 'r') as f:
		setting = json.load(f)

	# 修改模型名称映射逻辑，支持 LMFE-Net
	if MODEL == 'LMFE-Net':
		network = lmfe_net()
	else:
		network = eval(MODEL.replace('-', '_'))()
	network = nn.DataParallel(network).cuda()

	criterion = nn.L1Loss()

	if setting['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
	elif setting['optimizer'] == 'adamw':
		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
	else:
		raise Exception("ERROR: unsupported optimizer") 

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
	scaler = GradScaler()

	# 使用静态变量替代原来的 args.xxx
	dataset_dir = os.path.join(DATA_DIR, args.dataset)  # 保留数据集参数
	train_dataset = PairLoader(dataset_dir, 'train', 'train', 
								setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
	train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True,
                              drop_last=True)
	val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'], 
							  setting['patch_size'])
	val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=NUM_WORKERS,
                            pin_memory=True)

	# 使用静态变量替代原来的 args.xxx
	save_dir = os.path.join(SAVE_DIR, EXP)
	os.makedirs(save_dir, exist_ok=True)

	if not os.path.exists(os.path.join(save_dir, MODEL+'.pth')):
		print('==> Start training, current model name: ' + MODEL)
		# print(network)

		# 使用静态变量替代原来的 args.xxx
		writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, EXP, MODEL))

		best_psnr = 0
		for epoch in tqdm(range(setting['epochs'] + 1)):
			loss = train(train_loader, network, criterion, optimizer, scaler)

			writer.add_scalar('train_loss', loss, epoch)

			scheduler.step()

			if epoch % setting['eval_freq'] == 0:
				avg_psnr = valid(val_loader, network)
				
				writer.add_scalar('valid_psnr', avg_psnr, epoch)

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					torch.save({'state_dict': network.state_dict()},
                			   os.path.join(save_dir, MODEL+'.pth'))
				
				writer.add_scalar('best_psnr', best_psnr, epoch)

	else:
		print('==> Existing trained model')
		exit(1)