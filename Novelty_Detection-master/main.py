import os
import argparse
import torch
import torchvision
import torchvision.transforms as tf
from train import train_model
from train_combined_trans import train_model_combined
from train_conv_ae import train_conv_ae
from train_no_discrim_conv_ae import train_recon_err_conv_ae
from analyze import analyze_model
from model import R_Net, D_Net, R_Loss, D_Loss, R_WLoss, D_WLoss, D_Loss_combined, R_Loss_combined, \
	Dataset, trans_r_net, trans_d_net, trans_d_net_combined, Recon_Loss
import numpy as np

def main(args):

	torch.manual_seed(0)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	# Uncomment this if HTTP error happened

	# new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
	# torchvision.datasets.MNIST.resources = [
	#    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
	#    for url, md5 in torchvision.datasets.MNIST.resources
	# ]
	
	train_raw_dataset = torchvision.datasets.MNIST(root='../mnist', 
									train=True, 
									download=False,
									transform=tf.Compose([tf.Resize((32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]))
	
	valid_raw_dataset = torchvision.datasets.MNIST(root='../mnist', 
									train=False, 
									download=False, 
									transform=tf.Compose([tf.Resize((32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]))
	# Train and validate only on pictures of 1
	train_dataset = Dataset(train_raw_dataset, [args.target_cls])
	valid_dataset = Dataset(valid_raw_dataset, [0,1,2,3,4,5,6,7,8,9])
	#valid_dataset = Dataset(valid_raw_dataset, [1])
	#indices = np.arange(0,len(valid_dataset))
	#valid_indices = torch.from_numpy(np.random.choice(indices, size=500, replace=False))
	#valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)
	print("Valid dataset size", len(valid_dataset))
	
	if args.gpu and torch.cuda.is_available():
		device = torch.device('cuda:0')
		print(f'Using GPU {torch.cuda.get_device_name()}')
		print(torch.cuda.get_device_properties(device))
	else:
		device = torch.device('cpu')
		print('Using CPU')
	
	if args.load_path:
		r_net_path = os.path.join(args.load_path, args.r_load_path)
		d_net_path = os.path.join(args.load_path, args.d_load_path)
		r_net = torch.load(r_net_path).to(device)
		print(f'Loaded R_Net from {r_net_path}')
		d_net = torch.load(d_net_path).to(device)
		print(f'Loaded D_Net from {d_net_path}')
	else:
		#r_net = R_Net(in_channels = 1, std = args.std, skip = args.res, cat = args.cat).to(device)
		#d_net = D_Net(in_resolution = (28, 28), in_channels = 1).to(device)
		if args.autoenc == 'transformer' or 'recon_err_only':
			r_net = trans_r_net(image_size=32, patch_size=4, dim=128, depth=3, heads=4, mlp_dim=64,
							channels=1, gen_depth1=5, gen_depth2=2, gen_depth3=2, gen_init_size=8,
							gen_dim=128, gen_heads=4, gen_mlp_ratio=4, gen_drop_rate=0.5).to(device)
		elif args.autoenc == 'convolutional':
			print("convolutional AE")
			r_net = R_Net(in_channels=1, std=args.std, skip=args.res, cat=args.cat).to(device)
		if args.disc == 'separate':
			d_net = trans_d_net(image_size=32, patch_size=4, dim=32,
								depth=3, heads=4, mlp_dim=32, channels=1,
								dim_disc_head=16).to(device)
			#d_net = D_Net(in_resolution=(28, 28), in_channels=1).to(device)
		elif args.disc == 'combined':
			d_net = trans_d_net_combined(image_size=32, patch_size=4, dim=128,
								depth=3, heads=4, mlp_dim=32, channels=1,
								dim_disc_head=16).to(device)
		print('Created models')
	
	# TRAINING PARAMETERS

	save_path = (args.save_path, args.r_save_path, args.d_save_path)
	optim_r_params = {'alpha' : 0.9, 'weight_decay' : 1e-9}
	optim_d_params = {'alpha' : 0.9, 'weight_decay' : 1e-9}

	if args.analyze != 'no':
		analyze_model(r_net, d_net, train_dataset, valid_dataset, R_Loss, D_Loss, optimizer_class=torch.optim.RMSprop,
						device=device, batch_size=args.batch_size, optim_r_params=optim_r_params, optim_d_params=optim_d_params,
						learning_rate=args.lr, rec_loss_bound=args.rec_bound,
						save_step=args.sstep, num_workers=args.nw, save_path=save_path, lambd=args.lambd, test_class=args.classes_to_analyze)
	else:
		if args.autoenc == 'recon_err_only':
			model = train_recon_err_conv_ae(r_net, d_net, train_dataset, valid_dataset, Recon_Loss, D_Loss,
								  optimizer_class=torch.optim.RMSprop,
								  device=device, batch_size=args.batch_size, optim_r_params=optim_r_params,
								  optim_d_params=optim_d_params,
								  learning_rate=args.lr, rec_loss_bound=args.rec_bound, max_epochs=args.epochs,
								  save_step=args.sstep, num_workers=args.nw, save_path=save_path, lambd=args.lambd,target_cls=args.target_cls)
		elif args.autoenc == 'convolutional':
			model = train_conv_ae(r_net, d_net, train_dataset, valid_dataset, R_Loss, D_Loss,
								optimizer_class=torch.optim.RMSprop,
								device=device, batch_size=args.batch_size, optim_r_params=optim_r_params,
								optim_d_params=optim_d_params,
								learning_rate=args.lr, rec_loss_bound=args.rec_bound, max_epochs=args.epochs,
								save_step=args.sstep, num_workers=args.nw, save_path=save_path, lambd=args.lambd,target_cls=args.target_cls)
		elif args.disc == 'separate':
			model = train_model(r_net, d_net, train_dataset, valid_dataset, R_Loss, D_Loss, optimizer_class=torch.optim.RMSprop,
							device=device, batch_size=args.batch_size, optim_r_params=optim_r_params, optim_d_params=optim_d_params,
							learning_rate=args.lr, rec_loss_bound=args.rec_bound, max_epochs=args.epochs,
							save_step=args.sstep, num_workers=args.nw, save_path=save_path, lambd=args.lambd,target_cls=args.target_cls)
		elif args.disc == 'combined':
			model = train_model_combined(r_net, d_net, train_dataset, valid_dataset, R_Loss_combined, D_Loss_combined,
								optimizer_class=torch.optim.RMSprop,
								device=device, batch_size=args.batch_size, optim_r_params=optim_r_params,
								optim_d_params=optim_d_params,
								learning_rate=args.lr, rec_loss_bound=args.rec_bound, max_epochs=args.epochs,
								save_step=args.sstep, num_workers=args.nw, save_path=save_path, lambd=args.lambd,target_cls=args.target_cls)

	#avg_auc = test_model()

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', action='store_false', help='Turns off gpu training')
	parser.add_argument('--save_path', '-sp', type=str, default='.', help='Path to a folder where metrics and models will be saved')
	parser.add_argument('--d_save_path', '-dsp', type=str, default='d_net.pth', help='Name of .pth file for d_net to be saved')
	parser.add_argument('--r_save_path', '-rsp', type=str, default='r_net.pth', help='Name of .pth file for r_net to be saved')
	parser.add_argument('--load_path', '-lp', default=None, help='Path to a folder from which models will be loaded')
	parser.add_argument('--d_load_path', '-dlp', type=str, default='d_net.pth', help='Name of .pth file for d_net to be loaded')
	parser.add_argument('--r_load_path', '-rlp', type=str, default='r_net.pth', help='Name of .pth file for r_net to be loaded')
	parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loader')
	parser.add_argument('--epochs', type=int, default=85, help='epochs to train for')
	parser.add_argument('--rec_bound', type=float, default=0.1, help='Upper bound of reconstruction loss')
	parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate')
	parser.add_argument('--nw', type=int, default=1, help='num_workers for DataLoader')
	parser.add_argument('--sstep', type=int, default=5, help='Step in epochs for saving models')
	parser.add_argument('--std', type=float, default=0.155, help='Standart deviation for noise in R_Net')
	parser.add_argument('--lambd', type=float, default=0.2, help='Lambda parameter for LR loss')
	parser.add_argument('--cat', action='store_true', help='Turns on skip connections with concatenation')
	parser.add_argument('--res', action='store_true', help='Turns on residual connections')
	parser.add_argument('--analyze', '-a', type=str, default='no', help='to analyze or not')
	parser.add_argument('--classes_to_analyze', '-cta', type=int, default=1, help='which mnist class to analyze')
	parser.add_argument('--autoenc', '-ae', type=str, default='transformer', help='What kind of autoencoder to use')
	parser.add_argument('--disc', '-dc', type=str, default='separate', help='What kind of discriminator to use')
	parser.add_argument('--target_cls', '-tcs', type=int, default=1, help='which mnist class to use as inlier for anomaly detection')
	args = parser.parse_args()

	main(args)
