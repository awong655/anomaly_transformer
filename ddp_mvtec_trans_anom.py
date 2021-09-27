import os
from collections import OrderedDict
from os import listdir
from os.path import isfile, join

from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image, make_grid
import torch.nn as nn
import torch.optim as optim
import time
import argparse
from PIL import Image
import torch.nn.functional as F

from mvtecad_pytorch.dataset import MVTecADDataset
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
import random

from autoenc.ae_model import tmpTransGan, facebook_vit
from trans_discrim.trans_discrim import ViT_Discrim
from trans_discrim.trans_enc import VisionTransformer


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


CLS2IDX = {'bottle': 0,
           'cable': 1,
           'capsule': 2,
           'carpet': 3,
           'grid': 4,
           'hazelnut': 5,
           'leather': 6,
           'metal_nut': 7,
           'pill': 8,
           'screw': 9,
           'tile': 10,
           'toothbrush': 11,
           'transistor': 12,
           'wood': 13,
           'zipper': 14, }

classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
           'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def print_top_classes(predictions, **kwargs):
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(CLS2IDX[cls_idx])
        if len(CLS2IDX[cls_idx]) > max_str_len:
            max_str_len = len(CLS2IDX[cls_idx])

    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, CLS2IDX[cls_idx])
        output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)

def get_discrim(d_net: torch.nn.Module, x_input: torch.Tensor, x_enc: torch.Tensor) -> torch.Tensor:
	pred = d_net(x_input, x_enc.detach())
	pred = pred.sum(dim=1, keepdim=True).squeeze()
	pred[pred > 1] = 1
	return pred

# from ALOC- modified
def discrim_loss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor,
                 x_real_enc: torch.Tensor) -> torch.Tensor:
    pred_real = d_net(x_real, x_real_enc.detach())
    pred_fake = d_net(x_fake.detach(), x_real_enc.detach())

    pred_real = pred_real.sum(dim=1, keepdim=True).squeeze()
    pred_real[pred_real > 1] = 1
    pred_fake = pred_fake.sum(dim=1, keepdim=True).squeeze()
    pred_fake[pred_fake > 1] = 1

    y_real = torch.zeros_like(pred_real)
    y_fake = torch.ones_like(pred_fake) # these are ones because anom = 1, normal = 0

    real_loss = F.binary_cross_entropy_with_logits(pred_real, y_real)
    fake_loss = F.binary_cross_entropy_with_logits(pred_fake, y_fake)

    return real_loss + fake_loss

def train(gpu, args):
    print("Cuda Device:", torch.cuda.current_device())
    print("Training and not testing")
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    for idx, cls in enumerate(classes):
        print(args.train_cls)
        print(cls)
        if cls not in args.train_cls:
            continue
        print('==> Building models..')

        '''
        encoder = facebook_vit(image_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=512,
                           depth=6, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                           drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, keep_head=False).get_ViT().cuda()
                           
        '''
        encoder = VisionTransformer(image_size=32, patch_size=4, dim=512,
                                    depth=6, heads=8, mlp_dim=512, channels=3).cuda()
        generator = tmpTransGan(depth1=5, depth2=2, depth3=2, initial_size=8, dim=512,
                              heads=8, mlp_ratio=4, drop_rate=0.5).get_TransGan().cuda()

        discriminator = ViT_Discrim(image_size=32, patch_size=4, dim=512,
                                    depth=6, heads=8, mlp_dim=512, channels=3,
                                    dim_disc_head=256).cuda()

        ae_criterion = nn.MSELoss()

        enc_optimizer = optim.Adam(encoder.parameters(), lr=3e-5)
        gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001, eps=1e-08)
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.001, eps=1e-08)

        _transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()])

        train_set = MVTecADDataset(root='./mvtec', target=cls, transforms=_transforms, mask_transforms=_transforms,
                                   train=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=30,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   pin_memory=True)
        loss_values = []
        anom_loss_values = []
        ae_loss_values = []
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            train_loss = 0
            running_anom_loss = 0.0
            running_ae_loss = 0.0
            for batch_idx, (inputs, masks, targets) in enumerate(train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                enc_optimizer.zero_grad()
                gen_optimizer.zero_grad()
                disc_optimizer.zero_grad()
                encodings = encoder(inputs)
                recons = generator(encodings[:,0]) # first element of each sequence of patches is cls embedding
                anom_loss = discrim_loss(discriminator, inputs, recons, encodings)
                ae_loss = ae_criterion(recons, inputs)
                loss = ae_loss + anom_loss
                loss.backward()
                enc_optimizer.step()
                gen_optimizer.step()
                disc_optimizer.step()


                train_loss += loss.item()
                running_anom_loss += anom_loss.item()
                running_ae_loss += ae_loss.item()
                # _, predicted = outputs.max(1)
                # total += targets.size(0)
                # correct += predicted.eq(targets).sum().item()
                if gpu == 0:
                    print("Epoch No. ", epoch, "Batch Index.", batch_idx, "_ae Loss: ", (ae_loss/(batch_idx + 1)), "_anom_loss_", (anom_loss/(batch_idx + 1)), "_total loss_", (train_loss /(batch_idx + 1)))

            if gpu == 0:
                # Track
                loss_values.append(train_loss/len(train_set))
                anom_loss_values.append(running_anom_loss / len(train_set))
                ae_loss_values.append(running_ae_loss / len(train_set))

        if gpu == 0:
            # Plotting all losses
            fig, axs = plt.subplots(3)
            fig.suptitle('Vertically stacked subplots')
            axs[0].set_title("Overall Training Loss over Epochs")
            axs[0].plot(loss_values)
            axs[1].set_title("Reconsutrction Loss over Epochs")
            axs[1].plot(ae_loss_values)
            axs[2].set_title("Anomaly Detection Loss over Epochs")
            axs[2].plot(anom_loss_values)
            plt.savefig("train_loss_plt.png")

            print("saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': enc_optimizer.state_dict(),
            }, "./checkpoint/ckpt_enc_" + classes[idx] + ".pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': gen_optimizer.state_dict(),
            }, "./checkpoint/ckpt_gen_" + classes[idx] + ".pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': disc_optimizer.state_dict(),
            }, "./checkpoint/ckpt_disc_" + classes[idx] + ".pth")
            print("Save complete.")



def test(epoch, cls, encoder, generator, discriminator, ae_criterion, testloader, loader_idx, device):
    with torch.no_grad():
        print("TESTING")
        global best_acc
        # net.eval()
        test_loss = 0
        correct = 0
        total = 0
        # with torch.no_grad():
        for batch_idx, (inputs, masks, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets[targets>0] = 1
            encodings = encoder(inputs)
            recons = generator(encodings[:,0]) # first element of each sequence of patches is cls embedding. Only use this to generate reconstruction
            anom_res = get_discrim(discriminator, inputs, encodings)
            anom_loss = discrim_loss(discriminator, inputs, recons, encodings)
            ae_loss = ae_criterion(recons, inputs)
            loss = ae_loss + anom_loss
            test_loss += loss.item()
            print("TEST: Epoch No. ", epoch, "Batch Index.", batch_idx, "Loss: ", (test_loss / (batch_idx + 1)))
            # if (batch_idx % 50 == 0):
            print("input_Inlier: " + classes[loader_idx] + "_epoch_" + str(
                epoch) + "_" + str(batch_idx) + "_anom_result_" + str(anom_res))
            print("target_Inlier: " + classes[loader_idx] + "_epoch_" + str(
                epoch) + "_" + str(batch_idx) + "_anom_result_" + str(targets))
            cpu_inp = inputs.cpu()
            cpu_recons = recons.cpu()
            cpu_errors = cpu_inp - cpu_recons
            save_image(make_grid(cpu_inp, nrows=10),
                       "./cifar_imgs/ae_recons/test_input_Inlier: " + classes[loader_idx] + "_epoch_" + str(
                           epoch) + "_" + str(batch_idx) + ".jpg")
            save_image(make_grid(cpu_recons, nrows=10),
                       "./cifar_imgs/ae_recons/test_recon_Inlier:" + classes[loader_idx] + "_epoch_" + str(
                           epoch) + "_" + str(batch_idx) + ".jpg")
            save_image(make_grid(cpu_errors, nrows=10),
                       "./cifar_imgs/ae_recons/test_err_Inlier:" + classes[loader_idx] + "_epoch_" + str(epoch) + "_" + str(
                           batch_idx) + ".jpg")

def load_models(checkpoint_directory):
    ckpt_enc = [f for f in listdir(checkpoint_directory) if isfile(join(checkpoint_directory, f)) and "enc" in f]
    ckpt_gen = [f for f in listdir(checkpoint_directory) if isfile(join(checkpoint_directory, f)) and "gen" in f]
    ckpt_disc = [f for f in listdir(checkpoint_directory) if isfile(join(checkpoint_directory, f)) and "disc" in f]
    ckpt_enc.sort()
    ckpt_gen.sort()
    ckpt_disc.sort()
    return zip(ckpt_enc, ckpt_gen, ckpt_disc)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print("DEVICE:", device)
    if args.train == 'yes':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'
        print("DEVICE:", device)

        args.world_size = args.gpus * args.nodes
        args.port = random.randint(49152, 65535)
        print(args.master_addr)
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = str(args.port)
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    else:
        for idx, cls in enumerate(classes):
            if args.test_cls != '':
                if cls != args.test_cls:
                    continue
            # Model
            print('==> Building testing models..')
            encoder = VisionTransformer(image_size=32, patch_size=4, dim=512,depth=6, heads=8, mlp_dim=512, channels=3).cuda()

            generator = tmpTransGan(depth1=5, depth2=2, depth3=2, initial_size=8, dim=512,
                                  heads=8, mlp_ratio=4, drop_rate=0.5).get_TransGan()
            discriminator = ViT_Discrim(image_size=32, patch_size=4, dim=512,
                                    depth=6, heads=8, mlp_dim=512, channels=3,
                                    dim_disc_head=256).cuda()

            ae_criterion = nn.MSELoss()

            _transforms = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()])

            train_set = MVTecADDataset(root='./mvtec', target=cls, transforms=_transforms, mask_transforms=_transforms, train=True)
            test_set = MVTecADDataset(root='./mvtec', target=cls, transforms=_transforms, mask_transforms=_transforms, train=False)

            train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                       batch_size=10,
                                                       shuffle=True,
                                                       num_workers=2,
                                                       pin_memory=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                      batch_size=5,
                                                      shuffle=False,
                                                      num_workers=2,
                                                      pin_memory=True)

            print("only testing")
            checkpoints = list(load_models(args.checkpoint_directory))
            tmp_dict = torch.load(args.checkpoint_directory + "/" + checkpoints[idx][0])['model_state_dict']
            enc_state_dict = OrderedDict()
            for k, v in tmp_dict.items():
                name = k[:]  # remove `module.`
                enc_state_dict[name] = v
            encoder.load_state_dict(enc_state_dict)
            encoder = encoder.to(device)
            print("loaded encoder")
            tmp_dict = torch.load(args.checkpoint_directory + "/" + checkpoints[idx][1])['model_state_dict']
            gen_state_dict = OrderedDict()
            for k, v in tmp_dict.items():
                name = k[:]  # remove `module.`
                gen_state_dict[name] = v
            generator.load_state_dict(gen_state_dict)
            generator = generator.to(device)
            print("loaded generator")
            tmp_dict = torch.load(args.checkpoint_directory + "/" + checkpoints[idx][2])['model_state_dict']
            disc_state_dict = OrderedDict()
            for k, v in tmp_dict.items():
                name = k[:]  # remove `module.`
                disc_state_dict[name] = v
            discriminator.load_state_dict(disc_state_dict)
            discriminator = discriminator.to(device)
            print("loaded discriminator")
            if device == 'cuda':
                encoder = torch.nn.DataParallel(encoder)
                generator = torch.nn.DataParallel(generator)
                discriminator = torch.nn.DataParallel(discriminator)
                cudnn.benchmark = True
            for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
                test(epoch, cls, encoder, generator, discriminator, ae_criterion, test_loader, idx, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ma', '--master_addr', default='', type=str,
                        help='Master Address of node')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--resume_enc', '-re', action='store_true',
                        help='resume encoder from checkpoint')
    parser.add_argument('--resume_dec', '-rd', action='store_true',
                        help='resume decoder from checkpoint')
    parser.add_argument('--train', default='yes', type=str,
                        help='Whether to train or to test')
    parser.add_argument('--checkpoint_directory', default='./checkpoint/ae/', type=str,
                        help='Which directory the checkpoints are stored in')
    parser.add_argument('--test_cls', default='', type=str,
                        help='What class to test')
    parser.add_argument('--train_cls', nargs="*", default=[],
                        help='List of classes to test')
    args = parser.parse_args()
    main(args)
