import os
from collections import OrderedDict
from os import listdir
from os.path import isfile, join

import torchvision
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, random_split, Subset
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

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print("DEVICE:", device)
    trainset = torchvision.datasets.MNIST('./mnist/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   transforms.Resize((32, 32)),
                                   torchvision.transforms.ToTensor()
                               ]))


    testset = torchvision.datasets.MNIST('./mnist/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   transforms.Resize((32, 32)),
                                   torchvision.transforms.ToTensor()
                               ]))

    allSampleSet = trainset + testset

    classes = ('0-zero','1-one','2-two','3-three','4-four','5-five','6-six','7-seven','8-eight','9-nine')

    train_inliers = [np.where(np.array(trainset.targets) == class_idx)[0]
                     for class_idx in trainset.class_to_idx.values()]
    train_outliers = [np.where(np.array(trainset.targets) != class_idx)[0]
                      for class_idx in trainset.class_to_idx.values()]
    test_inliers = [np.where(np.array(testset.targets) == class_idx)[0]
                    for class_idx in testset.class_to_idx.values()]
    test_outliers = [np.where(np.array(testset.targets) != class_idx)[0]
                     for class_idx in testset.class_to_idx.values()]

    for i in range(len(classes)):
        test_inliers[i] += len(trainset)
        test_outliers[i] += len(trainset)

        # Drop elements
        train_outliers[i] = np.random.choice(train_outliers[i], size=500, replace=False)
        test_outliers[i] = np.random.choice(test_outliers[i], size=500, replace=False)

    inliers_zip = zip(train_inliers, test_inliers)
    inliers = [np.concatenate((i, j), dtype=np.int64) for i, j in inliers_zip]

    for i in inliers:
        print("Inlier size: ", len(i))

    outliers_zip = zip(train_outliers, test_outliers)
    outliers = [np.concatenate((i, j), dtype=np.int64) for i, j in outliers_zip]

    for i in outliers:
        print("Outlier size: ", len(i))

    trainloader = [
        DataLoader(
            dataset=Subset(allSampleSet, inds),
            batch_size=5,
            shuffle=True,
            num_workers=2
        ) for inds in inliers]

    testloader = [
        DataLoader(
            dataset=Subset(allSampleSet, inds),
            batch_size=50,
            shuffle=True,
            num_workers=2
        ) for inds in outliers]

    unified_loaders = list(zip(trainloader, testloader))

    for idx, loaders in enumerate(unified_loaders):

        # Model
        print('==> Building models..')
        encoder = VisionTransformer(image_size=32, patch_size=4, dim=256, depth=3, heads=4, mlp_dim=256,
                                    channels=1)

        generator = tmpTransGan(depth1=5, depth2=2, depth3=2, initial_size=8, dim=256,
                                heads=4, mlp_ratio=4, drop_rate=0.5).get_TransGan()
        discriminator = ViT_Discrim(image_size=32, patch_size=4, dim=256,
                                    depth=3, heads=4, mlp_dim=256, channels=1,
                                    dim_disc_head=128)
        encoder = encoder.to(device)
        generator = generator.to(device)
        discriminator = discriminator.to(device)

        ae_criterion = nn.MSELoss()

        enc_optimizer = optim.Adam(encoder.parameters(), lr=3e-5)
        gen_optimizer = optim.Adam(generator.parameters(), lr=0.00001, eps=1e-08)
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, eps=1e-08)

        '''
        if args.resume_enc:
            # Load checkpoint.
            print('==> Resuming encoder from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint_encoder = torch.load('./checkpoint/ckpt_enc.pth')
            encoder.load_state_dict(checkpoint_encoder['encoder'])
            decoder.load_state_dict(checkpoint_encoder['decoder'])
            start_epoch = checkpoint_encoder['epoch']

        if args.resume_dec:
            # Load checkpoint.
            print('==> Resuming decoder from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint_decoder = torch.load('./checkpoint/ckpt_dec.pth')
            encoder.load_state_dict(checkpoint_encoder['encoder'])
            decoder.load_state_dict(checkpoint_encoder['decoder'])
            start_epoch = checkpoint_decoder['epoch']
        '''

        print("Training inlier class ", classes[idx])
        for epoch in range(args.start_epoch, args.start_epoch + 200):
            train(epoch, loaders[0], enc_optimizer, gen_optimizer, disc_optimizer, encoder, generator, discriminator)
            test(epoch, loaders[1], idx, encoder, generator, discriminator, ae_criterion, classes)

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        print("saving model...")
        torch.save({
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': enc_optimizer.state_dict(),
        }, "./checkpoint/ckpt_enc_" + classes[idx] + ".pth")
        torch.save({
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': gen_optimizer.state_dict(),
        }, "./checkpoint/ckpt_gen_" + classes[idx] + ".pth")
        torch.save({
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': disc_optimizer.state_dict(),
        }, "./checkpoint/ckpt_disc_" + classes[idx] + ".pth")
        print("Save complete.")


def get_discrim(d_net: torch.nn.Module, x_input: torch.Tensor, x_enc: torch.Tensor):
    anom_pred, patch_pred = d_net(x_input, x_enc.detach())
    print("ANOM_PRED", torch.sigmoid(anom_pred))
    # print("PATCH_PRED", patch_pred)
    anom_pred = torch.round(torch.sigmoid(anom_pred))
    patch_pred = torch.round(patch_pred)
    # anom_pred = pred.sum(dim=1, keepdim=True).squeeze()
    # #anom_pred[anom_pred > 1] = 1
    return anom_pred, patch_pred


# from ALOC- modified
def discrim_loss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor,
                 x_real_enc: torch.Tensor) -> torch.Tensor:
    pred_real, _ = d_net(x_real, x_real_enc.detach())
    pred_fake, _ = d_net(x_fake.detach(), x_real_enc.detach())  # new leaf in graph because these values are
    # calculated by the other model.

    # print("PRED REAL OUTPUT: ", pred_real)
    # print("PRED FAKE OUTPUT: ", pred_fake)

    # pred_real = pred_real.sum(dim=1, keepdim=True).squeeze()
    # pred_real[pred_real > 1] = 1
    # pred_fake = pred_fake.sum(dim=1, keepdim=True).squeeze()
    # pred_fake[pred_fake > 1] = 1

    y_real = torch.zeros_like(pred_real)
    y_fake = torch.ones_like(pred_fake)  # these are ones because anom = 1, normal = 0
    # print("REAL PRED", pred_real)
    # print("FAKE PRED", pred_fake)
    real_loss = F.binary_cross_entropy_with_logits(pred_real, y_real)
    fake_loss = F.binary_cross_entropy_with_logits(pred_fake, y_fake)

    return real_loss + fake_loss


def r_loss(d_net, x_real, x_fake, x_real_enc, lamb):
    anom_pred, patch_pred = d_net(x_fake, x_real_enc)
    y = torch.ones_like(anom_pred)

    rec_loss = F.mse_loss(x_fake, x_real)
    gen_loss = F.binary_cross_entropy_with_logits(anom_pred, y)  # Generator loss

    L_r = gen_loss + lamb * rec_loss

    return rec_loss, gen_loss, L_r

def train(epoch, trainloader, enc_optimizer, gen_optimizer, disc_optimizer, encoder, generator, discriminator):
    print('\nEpoch: %d' % epoch)
    # net.train()
    running_ae_loss = 0
    running_generator_loss = 0.0
    running_disc_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.cuda(), targets.cuda()
        print(inputs.shape)
        enc_optimizer.zero_grad()
        gen_optimizer.zero_grad()
        disc_optimizer.zero_grad()

        encodings = encoder(inputs)
        recons = generator(encodings[:, 0])  # first element of each sequence of patches is cls embedding
        print("Reconstruction Shape", recons.shape)
        anom_loss = discrim_loss(discriminator, inputs, recons, encodings)
        anom_loss.backward()
        disc_optimizer.step()

        ae_loss, generator_loss, recon_loss = r_loss(discriminator, inputs, recons, encodings, 0.2)
        # ae_loss = ae_criterion(recons, inputs)
        # loss = ae_loss + anom_loss
        recon_loss.backward()
        enc_optimizer.step()
        gen_optimizer.step()

        running_ae_loss += ae_loss.item()
        running_generator_loss += generator_loss.item()
        running_disc_loss += anom_loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        print("Epoch No. ", epoch, "Batch Index.", batch_idx, "_ae Loss: ", (running_ae_loss / (batch_idx + 1)),
              "_gen_loss_", (running_generator_loss / (batch_idx + 1)), "_disc loss_",
              (running_disc_loss / (batch_idx + 1)))

def test(epoch, testloader, loader_idx, encoder, generator, discriminator, ae_criterion, classes):
    with torch.no_grad():
        print("TESTING")
        global best_acc
        # net.eval()
        test_loss = 0
        correct = 0
        total = 0
        # with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            targets[targets > 0] = 1
            encodings = encoder(inputs)
            recons = generator(encodings[:,
                               0])  # first element of each sequence of patches is cls embedding. Only use this to generate reconstruction
            anom_pred, anom_patch_pred = get_discrim(discriminator, inputs, encodings)
            anom_loss = discrim_loss(discriminator, inputs, recons, encodings)
            ae_loss = ae_criterion(recons, inputs)
            loss = ae_loss + anom_loss
            test_loss += loss.item()
            print("TEST: Epoch No. ", epoch, "Batch Index.", batch_idx, "Loss: ", (test_loss / (batch_idx + 1)))
            # if (batch_idx % 50 == 0):
            print("input_Inlier: " + classes[loader_idx] + "_epoch_" + str(
                epoch) + "_" + str(batch_idx) + "_anom_result_" + str(anom_pred))
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
                       "./cifar_imgs/ae_recons/test_err_Inlier:" + classes[loader_idx] + "_epoch_" + str(
                           epoch) + "_" + str(
                           batch_idx) + ".jpg")



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
