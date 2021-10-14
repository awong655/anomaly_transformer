import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve
import cv2
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def analyze_model(r_net: torch.nn.Module,
                d_net: torch.nn.Module,
                train_dataset: torch.utils.data.Dataset,
                valid_dataset: torch.utils.data.Dataset,
                r_loss,
                d_loss,
                lr_scheduler=None,
                optimizer_class=torch.optim.Adam,
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
                save_path: tuple = ('.', 'r_net.pth', 'd_net.pth'),
                test_class = 1) -> tuple:
    print("TESTINGGGGG**************************************")
    model_path = os.path.join(save_path[0], 'models')
    metric_path = os.path.join(save_path[0], 'metrics')
    r_net_path = os.path.join(model_path, save_path[1])
    d_net_path = os.path.join(model_path, save_path[2])

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)

    print(f'Models will be saved in {r_net_path} and {d_net_path}')
    print(f'Metrics will be saved in {metric_path}')

    optim_r = optimizer_class(r_net.parameters(), lr=learning_rate, **optim_r_params)
    optim_d = optimizer_class(d_net.parameters(), lr=learning_rate, **optim_d_params)

    if lr_scheduler:
        scheduler_r = lr_scheduler(optim_r, **scheduler_r_params)
        scheduler_d = lr_scheduler(optim_d, **scheduler_d_params)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, pin_memory=pin_memory,
                                               num_workers=num_workers)

    metrics = {'train': {'rec_loss': [], 'gen_loss': [], 'dis_loss': []},
               'valid': {'rec_loss': [], 'gen_loss': [], 'dis_loss': [],
                         'auc': [], 'patch_auc': []}}

    for epoch in range(max_epochs):

        start = timer()

        valid_metrics = validate_single_epoch(r_net, d_net, r_loss, d_loss, valid_loader, device, epoch, test_class)
        time = timer() - start

        #metrics['valid']['rec_loss'].append(valid_metrics['rec_loss'])
        #metrics['valid']['gen_loss'].append(valid_metrics['gen_loss'])
        #metrics['valid']['dis_loss'].append(valid_metrics['dis_loss'])
        metrics['valid']['auc'].append(valid_metrics['auc'])
        metrics['valid']['patch_auc'].append(valid_metrics['patch_auc'])

        if epoch % epoch_step == 0:
            print(f'Epoch {epoch}:')
            print("************************************************************************")
            print('VALID METRICS:', valid_metrics)
            print("Highest AUC: ", max(metrics['valid']['auc']))
            highestAUCIndex = metrics['valid']['auc'].index(max(metrics['valid']['auc']))
            print("Highest AUC Epoch: ", highestAUCIndex)

            print("Highest patch AUC: ", max(metrics['valid']['patch_auc']))
            highestAUCIndex = metrics['valid']['patch_auc'].index(max(metrics['valid']['patch_auc']))
            print("Highest AUC Epoch: ", highestAUCIndex)

            print(f'TIME: {time:.2f} s')

    #plot_learning_curves(metrics, metric_path)

    return (r_net, d_net)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def get_patches(imgs, patch_size):
    patch_height, patch_width = pair(patch_size)
    return rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)

def patch_to_img(patches, patch_size):
    patch_height, patch_width = pair(patch_size)
    return rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_height, p2=patch_width, h=8, w=8)

# patch predictions are a sequence of 0s and 1s, 0 = normal, 1 = anomaly
def get_patch_visualizations(d_net, x_input):
    real_pred, patch_pred = d_net(x_input)
    patches = get_patches(x_input, 4)
    mask = torch.ones_like(patches)
    #print("patches shape", patches.shape)
    #print(torch.sigmoid(patch_pred))

    # normalize patch predictions to between 0 and 1
    patch_pred -= patch_pred.min(1, keepdim=True)[0]
    patch_pred /= patch_pred.max(1, keepdim=True)[0]
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

def auc_patch_pred(d_net, x, targets):
	#pred, patch_pred = d_net(x)

	#patch_pred = torch.round(torch.absolute(1 - torch.sigmoid(patch_pred))) # flip from 0 as anomaly to 1 as anomaly
	#final_pred = patch_pred.sum(dim=1, keepdim=True).squeeze()
	#final_pred[final_pred > 1] = 1

	#fpr, tpr, thresholds = roc_curve(final_pred.cpu().numpy(), targets.cpu().numpy())
	#pred_auc = auc(fpr, tpr)
	return 0

def auc_pred(d_net, x, targets):
    pred, patch_pred = d_net(x)

    pred = torch.round(torch.sigmoid(pred))

    fpr, tpr, thresholds = roc_curve(targets.cpu().numpy(), pred.cpu().numpy())
    pred_auc = auc(fpr, tpr)
    return pred_auc

def validate_single_epoch(r_net, d_net, r_loss, d_loss, valid_loader, device, epoch, test_class) -> dict:
    r_net.eval()
    d_net.eval()

    valid_metrics = {'rec_loss': 0, 'gen_loss': 0, 'dis_loss': 0, 'auc': 0, 'patch_auc': 0}

    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            targets = data[1]
            targets[targets != test_class] = 0

            x = data[0].to(device)
            x_recon = r_net(x)
            #x_real_anomalies = get_patch_visualizations(d_net, x_real)
            x_anomalies = get_patch_visualizations(d_net, x)
            # print("Real Prediction", torch.sigmoid(d_net(x_real)))
            # print("Fake Prediction", torch.sigmoid(d_net(x_fake)))

            #dis_loss = d_loss(d_net, x_real, x_fake, do_print=False)

            #r_metrics = r_loss(d_net, x_real, x_fake, 0)



            #valid_metrics['rec_loss'] += r_metrics['rec_loss']
            #valid_metrics['gen_loss'] += r_metrics['gen_loss']
            #valid_metrics['dis_loss'] += dis_loss
            valid_metrics['auc'] += auc_pred(d_net, x, targets)
            valid_metrics['patch_auc'] += auc_patch_pred(d_net, x, targets)

            if epoch % 10 == 0:
                print(f'Saving test images on epoch {epoch}')
                # plot_learning_curves(metrics, metric_path)
                save_image(make_grid(x, nrows=10),
                           "../cifar_imgs/analysis/test_input_epoch_" + str(
                               epoch) + "_" + str(batch_idx) + ".jpg")
                save_image(make_grid(x_recon, nrows=10),
                           "../cifar_imgs/analysis/test_recon_epoch_" + str(
                               epoch) + "_" + str(batch_idx) + ".jpg")
                save_image(make_grid(x_anomalies, nrows=10),
                           "../cifar_imgs/analysis/analyze_anomalies" + str(
                               epoch) + "_" + str(batch_idx) + ".jpg")

    #valid_metrics['rec_loss'] = valid_metrics['rec_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
    #valid_metrics['gen_loss'] = valid_metrics['gen_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
    #valid_metrics['dis_loss'] = valid_metrics['dis_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
    valid_metrics['auc'] = valid_metrics['auc'] / (len(valid_loader.dataset) / valid_loader.batch_size)
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
    plt.plot(metrics['train']['rec_loss'], label='Train rec loss')
    plt.plot(metrics['valid']['rec_loss'], label='Dev rec loss')
    plt.title('Reconstruction loss evolution')
    plt.xlabel('epochs')
    plt.ylabel('Rec loss')
    plt.legend()
    plt.savefig(rec_loss_path)
    plt.close()

    # Plot discriminator loss: -log(D(x)) - log(1 - D(R(x')))
    plt.figure()
    plt.plot(metrics['train']['dis_loss'], label='Train dis loss')
    plt.plot(metrics['valid']['dis_loss'], label='Dev dis loss')
    plt.title('Discriminator loss evolution')
    plt.xlabel('epochs')
    plt.ylabel('Dis loss')
    plt.legend()
    plt.savefig(dis_loss_path)
    plt.close()

    # Plot generator loss: -log(D(R(x)))
    plt.figure()
    plt.plot(metrics['train']['gen_loss'], label='Train gen loss')
    plt.plot(metrics['valid']['gen_loss'], label='Dev gen loss')
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
