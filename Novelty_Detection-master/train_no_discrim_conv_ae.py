import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve, roc_auc_score
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def train_recon_err_conv_ae(r_net: torch.nn.Module,
						 d_net: torch.nn.Module,
						 train_dataset: torch.utils.data.Dataset,
						 valid_dataset: torch.utils.data.Dataset,
						 r_loss,
						 d_loss,
						 lr_scheduler = None,
						 optimizer_class = torch.optim.Adam,
						 optim_r_params: dict = {},
						 optim_d_params: dict = {},
						 learning_rate: float = 0.001,
						 scheduler_r_params: dict = {},
						 scheduler_d_params: dict = {},
						 batch_size: int = 512,
						 pin_memory: bool = True,
						 num_workers: int = 1,
						 max_epochs: int = 85,
						 epoch_step: int = 1,
						 save_step: int = 5,
						 rec_loss_bound: float = 0.1,
						 lambd: float = 0.2,
						 device: torch.device = torch.device('cpu'),
						 save_path: tuple = ('.','r_net.pth','d_net.pth'),
							target_cls = 1) -> tuple:

	model_path = os.path.join(save_path[0], 'models')
	metric_path= os.path.join(save_path[0], 'metrics')
	r_net_path = os.path.join(model_path, save_path[1])
	d_net_path = os.path.join(model_path, save_path[2])

	if not os.path.exists(model_path):
		os.makedirs(model_path)
	if not os.path.exists(metric_path):
		os.makedirs(metric_path)

	print(f'Models will be saved in {r_net_path} and {d_net_path}')
	print(f'Metrics will be saved in {metric_path}')

	optim_r = optimizer_class(r_net.parameters(), lr = learning_rate, **optim_r_params)
	optim_d = optimizer_class(d_net.parameters(), lr = 0.001, **optim_d_params)

	if lr_scheduler:
		scheduler_r = lr_scheduler(optim_r, **scheduler_r_params)
		scheduler_d = lr_scheduler(optim_d, **scheduler_d_params)

	train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=True, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)

	metrics =  {'train' : {'rec_loss' : [], 'gen_loss' : [], 'dis_loss' : [], 'auc' : []},
				'valid' : {'rec_loss' : [], 'gen_loss' : [], 'dis_loss' : [],
						   'auc' : [], 'patch_auc' : []}}

	for epoch in range(max_epochs):

		start = timer()
		train_metrics = train_single_epoch(r_net, d_net, optim_r, optim_d, r_loss, d_loss, train_loader, lambd, device, epoch)
		valid_metrics = validate_single_epoch_recon(r_net, d_net, r_loss, d_loss, valid_loader, device, epoch, test_class=1)
		time = timer() - start

		metrics['train']['rec_loss'].append(train_metrics['rec_loss'])
		metrics['train']['gen_loss'].append(train_metrics['gen_loss'])
		metrics['train']['dis_loss'].append(train_metrics['dis_loss'])
		metrics['train']['auc'].append(train_metrics['auc'])
		metrics['valid']['rec_loss'].append(valid_metrics['rec_loss'])
		metrics['valid']['gen_loss'].append(valid_metrics['gen_loss'])
		metrics['valid']['dis_loss'].append(valid_metrics['dis_loss'])
		metrics['valid']['auc'].append(valid_metrics['auc'])
		metrics['valid']['patch_auc'].append(valid_metrics['patch_auc'])

		if epoch % epoch_step == 0:
			print(f'Epoch {epoch}:')
			print("************************************************************************")
			print('TRAIN METRICS:', train_metrics)
			print('VALID METRICS:', valid_metrics)
			print("Highest Test AUC: ", max(metrics['valid']['auc']))
			highestAUCIndex = metrics['valid']['auc'].index(max(metrics['valid']['auc']))
			print("Highest Test AUC Epoch: ", highestAUCIndex)

			print("Highest Test patch AUC: ", max(metrics['valid']['patch_auc']))
			highestAUCIndex = metrics['valid']['patch_auc'].index(max(metrics['valid']['patch_auc']))
			print("Highest Test Patch AUC Epoch: ", highestAUCIndex)

			print(f'TIME: {time:.2f} s')

		if lr_scheduler:
			scheduler_r.step()
			scheduler_d.step()

		if epoch % save_step == 0:
			if highestAUCIndex == epoch:
				torch.save(r_net, r_net_path)
				torch.save(d_net, d_net_path)

		if valid_metrics['rec_loss'] < rec_loss_bound and train_metrics['rec_loss'] < rec_loss_bound:
			torch.save(r_net, r_net_path)
			torch.save(d_net, d_net_path)
			print('Reconstruction loss achieved optimum')
			print('Stopping training')

			break

	plot_learning_curves(metrics, metric_path)

	return (r_net, d_net)

def train_auc_pred(d_net, r_net, x_real, x_fake):
	real_pred, patch_pred_real = d_net(x_real)
	fake_pred, patch_pred_fake = d_net(x_fake)

	#print("train real pred", real_pred)
	#print("train fake pred", fake_pred)
	#print("real prediction shape", real_pred.shape)
	#print("real prediction normalized shape", normalize_matrix_rows(real_pred).shape)
	real_pred = torch.round(real_pred)
	fake_pred = torch.round(fake_pred)
	y_real = torch.ones_like(real_pred)
	y_fake = torch.zeros_like(real_pred)
	all_pred = torch.cat((real_pred, fake_pred))
	#print("NON PATCH PRED", all_pred)
	all_y = torch.cat((y_real, y_fake))
	#print("NON PATCH TARGETS", all_y)
	fpr, tpr, thresholds = roc_curve(all_y.cpu().numpy(), all_pred.cpu().numpy())
	pred_auc = auc(fpr, tpr)
	#pred_auc = roc_auc_score(all_y.cpu().numpy(), all_pred.cpu().numpy())
	return pred_auc


def train_single_epoch(r_net, d_net, optim_r, optim_d, r_loss, d_loss, train_loader, lambd, device, epoch) -> dict:

	r_net.train()
	d_net.train()

	train_metrics = {'rec_loss' : 0, 'gen_loss' : 0, 'dis_loss' : 0, 'auc' : 0}

	for batch_idx, data in enumerate(train_loader):

		r_net.zero_grad()
		x_real = data[0].to(device)
		_, x_fake = r_net(x_real)

		r_metrics = r_loss(x_real, x_fake) # L_r = gen_loss + lambda * rec_loss

		r_metrics.backward()
		optim_r.step()

		train_metrics['rec_loss'] += r_metrics
		train_metrics['gen_loss'] += 1
		train_metrics['dis_loss'] += 1

		with torch.no_grad():
			train_metrics['auc'] += 1

		if epoch % 10 == 0:
			print(f'Saving train images on epoch {epoch}')
			#plot_learning_curves(metrics, metric_path)
			save_image(make_grid(x_real, nrows=10),
					   "../cifar_imgs/ae_recons/train_input_epoch_" + str(
						   epoch) + "_" + str(batch_idx) + ".jpg")
			save_image(make_grid(x_fake, nrows=10),
					   "../cifar_imgs/ae_recons/train_recon_epoch_" + str(
						   epoch) + "_" + str(batch_idx) + ".jpg")

	train_metrics['rec_loss'] = train_metrics['rec_loss'].item() / (len(train_loader.dataset) / train_loader.batch_size)
	train_metrics['gen_loss'] = train_metrics['gen_loss'] / (len(train_loader.dataset) / train_loader.batch_size)
	train_metrics['dis_loss'] = train_metrics['dis_loss'] / (len(train_loader.dataset) / train_loader.batch_size)
	train_metrics['auc'] = train_metrics['auc'] / (batch_idx+1)
	return train_metrics

def pair(t):
	return t if isinstance(t, tuple) else (t, t)

def get_patches(imgs, patch_size):
	patch_height, patch_width = pair(patch_size)
	return rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)

def patch_to_img(patches, patch_size):
	patch_height, patch_width = pair(patch_size)
	return rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_height, p2=patch_width, h=8, w=8)

# patch predictions are a sequence of 0s and 1s, 0 = normal, 1 = anomaly
def get_patch_visualizations(d_net, r_net, x_input):
	real_pred, patch_pred = d_net(x_input)
	patches = get_patches(x_input, 4)
	mask = torch.ones_like(patches)
	#print("patches shape", patches.shape)
	#print(torch.sigmoid(patch_pred))

	# normalize patch predictions to between 0 and 1
	#patch_pred -= patch_pred.min(1, keepdim=True)[0]
	#patch_pred /= patch_pred.max(1, keepdim=True)[0]
	#print("patch pred shape", patch_pred.shape)

	# get patch prediction
	#patch_pred = torch.round(torch.absolute(1 - torch.sigmoid(patch_pred)))  # flip from 0 as anomaly to 1 as anomaly

	#print(patch_pred)

	# expand last dim of patch prediction to make room for expanded dim
	patch_pred = patch_pred.unsqueeze(-1)

	# copy value of last dim (anomaly score) and repeat to allow matmul with image patch tensor
	patch_pred = patch_pred.expand(-1, -1, 16)

	anom_vis = torch.mul(mask, patch_pred)

	# rearrange modified image back into rectangle
	anom_vis = patch_to_img(anom_vis, 4)
	return anom_vis

def auc_patch_pred(d_net, r_net, x, targets):
	'''
	pred, patch_pred = d_net(x)

	patch_pred = torch.round(torch.absolute(1 - torch.sigmoid(patch_pred))) # flip from 0 as anomaly to 1 as anomaly
	final_pred = patch_pred.sum(dim=1, keepdim=True).squeeze()
	final_pred[final_pred > 1] = 1
	'''
	#fpr, tpr, thresholds = roc_curve(final_pred.cpu().numpy(), targets.cpu().numpy())
	#pred_auc = auc(fpr, tpr)
	return 0

def auc_pred(d_net, r_net, x, targets):
	pred, patch_pred = d_net(x)
	pred = torch.round(pred)

	fpr, tpr, thresholds = roc_curve(targets.cpu().numpy(), pred.cpu().numpy())
	pred_auc = auc(fpr, tpr)
	#pred_auc = roc_auc_score(targets.cpu().numpy(), pred.cpu().numpy())
	return pred_auc

def validate_single_epoch_recon(r_net, d_net, r_loss, d_loss, valid_loader, device, epoch, test_class=1) -> dict:
	r_net.eval()
	d_net.eval()

	valid_metrics = {'rec_loss': 0, 'gen_loss': 0, 'dis_loss': 0, 'auc': 0, 'patch_auc': 0}

	with torch.no_grad():
		for batch_idx, data in enumerate(valid_loader):
			targets = data[1]
			targets[targets == test_class] = -1
			targets[targets >= 0] = 0
			targets[targets < 0] = 1
			x = data[0].to(device)
			torch_x = x
			_, x_recon = r_net(x)
			torch_recons = x_recon

			# torch to tensorflow
			targets = targets.cpu().numpy()
			targets = tf.convert_to_tensor(targets)
			x_recon = x_recon.cpu().numpy()
			x_recon = tf.convert_to_tensor(x_recon)
			x = x.cpu().numpy()
			x = tf.convert_to_tensor(x)

			# Get Normal Reconstruction Anomaly Score
			m1 = tf.keras.metrics.AUC(num_thresholds=100)
			gen_l2_loss_normal_not_reduced = tf.abs(x - x_recon) ** 2
			reconstruction_loss_normal_reduced = tf.map_fn(fn=lambda z: tf.reduce_sum(z),
														   elems=gen_l2_loss_normal_not_reduced) * -1
			min_normal = tf.reduce_min(reconstruction_loss_normal_reduced)
			max_normal = tf.reduce_max(reconstruction_loss_normal_reduced)
			reconstruction_loss_normal_normalized = tf.math.add(
				tf.math.divide(tf.math.subtract(reconstruction_loss_normal_reduced, max_normal),
							   tf.math.subtract(max_normal, min_normal)), 1)
			m1.update_state(targets, reconstruction_loss_normal_normalized)
			auc = tf.keras.backend.get_value(m1.result())

			print("**********************************************recon loss", reconstruction_loss_normal_normalized)

			valid_metrics['auc'] += auc
			valid_metrics['patch_auc'] += 0

			if epoch % 10 == 0:
				print(f'Saving test images on epoch {epoch}')
				# plot_learning_curves(metrics, metric_path)
				save_image(make_grid(torch_x, nrows=10),
						   "../cifar_imgs/analysis/test_input_epoch_" + str(
							   epoch) + "_" + str(batch_idx) + ".jpg")
				save_image(make_grid(torch_recons, nrows=10),
						   "../cifar_imgs/analysis/test_recon_epoch_" + str(
							   epoch) + "_" + str(batch_idx) + ".jpg")

	# valid_metrics['rec_loss'] = valid_metrics['rec_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
	# valid_metrics['gen_loss'] = valid_metrics['gen_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
	# valid_metrics['dis_loss'] = valid_metrics['dis_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
	valid_metrics['auc'] = valid_metrics['auc'] / (batch_idx+1)
	valid_metrics['patch_auc'] = valid_metrics['patch_auc'] / (len(valid_loader.dataset) / valid_loader.batch_size)

	return valid_metrics

def plot_learning_curves(metrics: dict, metric_path: str):

	rec_loss_path = os.path.join(metric_path, 'rec_loss.jpg')
	dis_loss_path = os.path.join(metric_path, 'dis_loss.jpg')
	gen_loss_path = os.path.join(metric_path, 'gen_loss.jpg')
	aucs_path = os.path.join(metric_path, 'auc.jpg')
	patch_auc_path = os.path.join(metric_path, 'patch_auc.jpg')

	# Plot reconstruction loss: ||X' - X||^2
	plt.figure()
	plt.plot(metrics['train']['rec_loss'], label = 'Train rec loss')
	plt.plot(metrics['valid']['rec_loss'], label = 'Dev rec loss')
	plt.title('Reconstruction loss evolution')
	plt.xlabel('epochs')
	plt.ylabel('Rec loss')
	plt.legend()
	plt.savefig(rec_loss_path)
	plt.close()

	# Plot discriminator loss: -log(D(x)) - log(1 - D(R(x')))
	plt.figure()
	plt.plot(metrics['train']['dis_loss'], label = 'Train dis loss')
	plt.plot(metrics['valid']['dis_loss'], label = 'Test dis loss')
	plt.title('Discriminator loss evolution')
	plt.xlabel('epochs')
	plt.ylabel('Dis loss')
	plt.legend()
	plt.savefig(dis_loss_path)
	plt.close()

	# Plot generator loss: -log(D(R(x)))
	plt.figure()
	plt.plot(metrics['train']['gen_loss'], label = 'Train gen loss')
	plt.plot(metrics['valid']['gen_loss'], label = 'Test gen loss')
	plt.title('Generator loss evolution')
	plt.xlabel('epochs')
	plt.ylabel('Gen loss')
	plt.legend()
	plt.savefig(aucs_path)
	plt.close()

	# Plot AUC
	plt.figure()
	plt.plot(metrics['valid']['auc'], label='AUC')
	plt.title('Anomaly Detection AUC')
	plt.xlabel('epochs')
	plt.ylabel('AUC')
	plt.legend()
	plt.savefig(gen_loss_path)
	plt.close()

	# Plot Patch Based Anomaly Detection AUC
	plt.figure()
	plt.plot(metrics['valid']['patch_auc'], label='Patch Based AUC')
	plt.title('Patch Anomaly Detection AUC')
	plt.xlabel('epochs')
	plt.ylabel('Patch AUC')
	plt.legend()
	plt.savefig(patch_auc_path)
	plt.close()
