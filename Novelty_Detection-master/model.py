import torch
import torch.nn.functional as F

from autoenc.ae_model import tmpTransGan, facebook_vit
from trans_discrim.trans_discrim_seperate import ViT_Discrim
from trans_discrim.trans_discrim import ViT_Discrim as ViT_Discrim_Combined
from trans_discrim.trans_enc import VisionTransformer

class Dataset(torch.utils.data.Dataset):

	def __init__(self, dataset: torch.utils.data.Dataset, labels: list):
		self.dataset = dataset
		self.labels = labels
		self.indexes = self._extract_indexes()

	def _extract_indexes(self):

		indexes = []

		for label in self.labels:
			for i, sample in enumerate(self.dataset):
				if sample[1] == label:
					indexes.append(i)
		print(f'Created dataset of length {len(indexes)} for labels: {self.labels}')
		return indexes

	def __len__(self):
		return len(self.indexes)
	
	def __getitem__(self, idx: int):
		return self.dataset[self.indexes[idx]]

class ModuleList(torch.nn.ModuleList):

	def forward(self, x:torch.Tensor, res_connections:list=[], cat:bool=False):
		connections = []
		for i, module in enumerate(self):
			if i % 3 == 0 and not bool(res_connections):
				x = module(x)
				connections.append(x)
			elif i % 3 == 0 and bool(res_connections):
				x = module(torch.cat((x, res_connections[i//3]), dim = 1)) if cat else module(x + res_connections[i//3])
			else:
				x = module(x)
		return x, connections

class trans_r_net(torch.nn.Module):
	def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels,
				 gen_depth1, gen_depth2, gen_depth3, gen_init_size, gen_dim, gen_heads, gen_mlp_ratio,
				 gen_drop_rate, std = 1.0):
		super(trans_r_net, self).__init__()
		self.std = std
		self.encoder = VisionTransformer(image_size=image_size, patch_size=patch_size, dim=dim,
									depth=depth, heads=heads, mlp_dim=mlp_dim, channels=channels)
		self.generator = tmpTransGan(depth1=gen_depth1, depth2=gen_depth2, depth3=gen_depth3,
									 initial_size=gen_init_size, dim=gen_dim,
								heads=gen_heads, mlp_ratio=gen_mlp_ratio, drop_rate=gen_drop_rate).get_TransGan()

	def forward(self, x:torch.Tensor, noise:bool = True):
		x_hat = self.add_noise(x) if noise else x
		z = self.encoder.forward(x_hat)

		x_out = self.generator.forward(z[:,0])

		return z, x_out

	def add_noise(self, x):

		noise = torch.randn_like(x) * self.std
		x_hat = x + noise

		return x_hat

class trans_d_net_combined(torch.nn.Module):
	def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels, dim_disc_head):
		super(trans_d_net_combined, self).__init__()
		self.discriminator = ViT_Discrim_Combined(image_size=image_size, patch_size=patch_size, dim=dim,
									depth=depth, heads=heads, mlp_dim=mlp_dim, channels=channels,
									dim_disc_head=dim_disc_head)

	def forward(self, x: torch.Tensor, x_enc: torch.Tensor):
		return self.discriminator(x, x_enc)

class trans_d_net(torch.nn.Module):
	def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels, dim_disc_head):
		super(trans_d_net, self).__init__()
		self.discriminator = ViT_Discrim(image_size=image_size, patch_size=patch_size, dim=dim,
									depth=depth, heads=heads, mlp_dim=mlp_dim, channels=channels,
									dim_disc_head=dim_disc_head)

	def forward(self, x: torch.Tensor):
		return self.discriminator(x)

class R_Net(torch.nn.Module):

	def __init__(self, activation = torch.nn.LeakyReLU, in_channels:int = 3, n_channels:int = 64,
					kernel_size:int = 5, std:float = 1.0, skip:bool = False, cat:bool = False):

		super(R_Net, self).__init__()

		self.activation = activation
		self.in_channels = in_channels
		self.n_c = n_channels
		self.k_size = kernel_size
		self.std = std
		self.cat = cat
		self.skip = skip if self.cat is False else False

		if self.skip:
			print('Turn on res connections')
		elif self.cat:
			print('Turn on concat skip')
		else:
			print('No skip connections')

		self.Encoder = ModuleList([torch.nn.Conv2d(self.in_channels, self.n_c, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c),
											self.activation(),
											torch.nn.Conv2d(self.n_c, self.n_c*2, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c*2),
											self.activation(),
											torch.nn.Conv2d(self.n_c*2, self.n_c*4, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c*4),
											self.activation(),
											torch.nn.Conv2d(self.n_c*4, self.n_c*8, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c*8),
											self.activation()])

		self.n_c = self.n_c * 2 if self.cat else self.n_c

		self.Decoder = ModuleList([torch.nn.ConvTranspose2d(self.n_c*8, n_channels*4, self.k_size, bias = False),
											torch.nn.BatchNorm2d(n_channels*4),
											self.activation(),
											torch.nn.ConvTranspose2d(self.n_c*4, n_channels*2, self.k_size, bias = False),
											torch.nn.BatchNorm2d(n_channels*2),
											self.activation(),
											torch.nn.ConvTranspose2d(self.n_c*2, n_channels, self.k_size, bias = False),
											torch.nn.BatchNorm2d(n_channels),
											self.activation(),
											torch.nn.ConvTranspose2d(self.n_c, self.in_channels, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.in_channels),
											self.activation()])

	def forward(self, x:torch.Tensor, noise:bool = True):

		x_hat = self.add_noise(x) if noise else x
		z, res_connections = self.Encoder.forward(x_hat)

		res_connections.reverse()

		if self.skip or self.cat:
			x_out, _ = self.Decoder.forward(z, res_connections, self.cat)
		else:
			x_out, _ = self.Decoder.forward(z)

		return x_out

	def add_noise(self, x):

		noise = torch.randn_like(x) * self.std
		x_hat = x + noise

		return x_hat

class D_Net(torch.nn.Module):

	def __init__(self, in_resolution:tuple, activation = torch.nn.LeakyReLU, in_channels:int = 3, n_channels:int = 64, kernel_size:int = 5):

		super(D_Net, self).__init__()

		self.activation = activation
		self.in_resolution = in_resolution
		self.in_channels = in_channels
		self.n_c = n_channels
		self.k_size = kernel_size

		self.cnn = torch.nn.Sequential(torch.nn.Conv2d(self.in_channels, self.n_c, self.k_size, bias = False),
										torch.nn.BatchNorm2d(self.n_c),
										self.activation(),
										torch.nn.Conv2d(self.n_c, self.n_c*2, self.k_size, bias = False),
										torch.nn.BatchNorm2d(self.n_c*2),
										self.activation(),
										torch.nn.Conv2d(self.n_c*2, self.n_c*4, self.k_size, bias = False),
										torch.nn.BatchNorm2d(self.n_c*4),
										self.activation(),
										torch.nn.Conv2d(self.n_c*4, self.n_c*8, self.k_size, bias = False),
										torch.nn.BatchNorm2d(self.n_c*8),
										self.activation())

		# Compute output dimension after conv part of D network

		self.out_dim = self._compute_out_dim()

		self.fc = torch.nn.Linear(self.out_dim, 1)

	def _compute_out_dim(self):
		
		test_x = torch.Tensor(1, self.in_channels, self.in_resolution[0], self.in_resolution[1])
		for p in self.cnn.parameters():
			p.requires_grad = False
		test_x = self.cnn(test_x)
		out_dim = torch.prod(torch.tensor(test_x.shape[1:])).item()
		for p in self.cnn.parameters():
			p.requires_grad = True

		return out_dim

	def forward(self, x:torch.Tensor):

		x = self.cnn(x)

		x = torch.flatten(x, start_dim = 1)

		out = self.fc(x)

		return out

def R_Loss_combined(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor, x_real_enc, lambd: float) -> dict:

	pred, _ = d_net(x_fake, x_real_enc)
	y = torch.zeros_like(pred)

	rec_loss = F.mse_loss(x_fake, x_real)
	gen_loss = F.binary_cross_entropy_with_logits(pred, y) # generator loss

	L_r = gen_loss + lambd * rec_loss

	return {'rec_loss' : rec_loss, 'gen_loss' : gen_loss, 'L_r' : L_r}

def D_Loss_combined(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor, x_real_enc, do_print=False) -> torch.Tensor:
	pred_real, _ = d_net(x_real, x_real_enc.detach())
	pred_fake, _ = d_net(x_fake.detach(), x_real_enc.detach())

	y_real = torch.zeros_like(pred_real)
	y_fake = torch.ones_like(pred_fake)

	real_loss = F.binary_cross_entropy_with_logits(pred_real, y_real)
	fake_loss = F.binary_cross_entropy_with_logits(pred_fake, y_fake)

	if do_print:
		print("REAL", pred_real.squeeze())
		print("REAL SIGMOID", torch.sigmoid(pred_real).squeeze())
		print("FAKE", pred_fake.squeeze())
		print("FAKE SIGMOID", torch.sigmoid(pred_fake).squeeze())

	return real_loss + fake_loss

def R_Loss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor, lambd: float) -> dict:

	pred, _ = d_net(x_fake)
	y = torch.ones_like(pred)

	rec_loss = F.mse_loss(x_fake, x_real)
	gen_loss = F.binary_cross_entropy_with_logits(pred, y) # generator loss

	L_r = gen_loss + lambd * rec_loss

	return {'rec_loss' : rec_loss, 'gen_loss' : gen_loss, 'L_r' : L_r}

def D_Loss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor, do_print=False) -> torch.Tensor:

	pred_real, _ = d_net(x_real)
	pred_fake, _ = d_net(x_fake.detach())
	
	y_real = torch.ones_like(pred_real)
	y_fake = torch.zeros_like(pred_fake)

	real_loss = F.binary_cross_entropy_with_logits(pred_real, y_real)
	fake_loss = F.binary_cross_entropy_with_logits(pred_fake, y_fake)

	if do_print:
		print("REAL", pred_real.squeeze())
		print("REAL SIGMOID", torch.sigmoid(pred_real).squeeze())
		print("FAKE", pred_fake.squeeze())
		print("FAKE SIGMOID", torch.sigmoid(pred_fake).squeeze())

	return real_loss + fake_loss

# Wasserstein GAN loss (https://arxiv.org/abs/1701.07875)

def R_WLoss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor, lambd: float) -> dict:

	pred = torch.sigmoid(d_net(x_fake))

	rec_loss = F.mse_loss(x_fake, x_real)
	gen_loss = -torch.mean(pred) # Wasserstein G loss: - E[ D(G(x)) ]

	L_r = gen_loss + lambd * rec_loss

	return {'rec_loss' : rec_loss, 'gen_loss' : gen_loss, 'L_r' : L_r}

def D_WLoss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:

	pred_real = torch.sigmoid(d_net(x_real))
	pred_fake = torch.sigmoid(d_net(x_fake.detach()))
	
	dis_loss = -torch.mean(pred_real) + torch.mean(pred_fake) # Wasserstein D loss: -E[D(x_real)] + E[D(x_fake)]

	return dis_loss
