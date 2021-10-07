import os
import torch
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve
import cv2
import numpy as np


def train_model(r_net: torch.nn.Module,
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
                save_path: tuple = ('.', 'r_net.pth', 'd_net.pth')) -> tuple:
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

        valid_metrics = validate_single_epoch(r_net, d_net, r_loss, d_loss, valid_loader, device, epoch)
        time = timer() - start

        metrics['valid']['rec_loss'].append(valid_metrics['rec_loss'])
        metrics['valid']['gen_loss'].append(valid_metrics['gen_loss'])
        metrics['valid']['dis_loss'].append(valid_metrics['dis_loss'])
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

def auc_patch_pred(d_net, x_real, x_fake):
    real_pred, patch_pred_real = d_net(x_real)
    fake_pred, patch_pred_fake = d_net(x_fake.detach())

    patch_pred_real = torch.round(torch.sigmoid(patch_pred_real))
    real_pred = patch_pred_real.sum(dim=1, keepdim=True).squeeze()
    real_pred[real_pred > 1] = 1

    print("PATCH PREDICTION REAL NO ROUND", patch_pred_real)

    patch_pred_fake = torch.round(torch.sigmoid(patch_pred_fake))
    fake_pred = patch_pred_fake.sum(dim=1, keepdim=True).squeeze()
    fake_pred[fake_pred > 1] = 1

    print("PATCH PREDICTION FAKE NO ROUND", fake_pred)

    y_real = torch.ones_like(real_pred)
    y_fake = torch.zeros_like(real_pred)
    all_pred = torch.cat((real_pred, fake_pred))
    print("PATCH PREDICTION", all_pred)
    all_y = torch.cat((y_real, y_fake))
    print("PATCH TARGETS", all_y)
    fpr, tpr, thresholds = roc_curve(all_y.cpu().numpy(), all_pred.cpu().numpy())
    pred_auc = auc(fpr, tpr)
    return pred_auc


# #anom_pred[anom_pred > 1] = 1

def auc_pred(d_net, x_real, x_fake):
    real_pred, patch_pred_real = d_net(x_real)
    fake_pred, patch_pred_fake = d_net(x_fake.detach())

    real_pred = torch.round(torch.sigmoid(real_pred).squeeze())
    fake_pred = torch.round(torch.sigmoid(fake_pred).squeeze())
    y_real = torch.ones_like(real_pred)
    y_fake = torch.zeros_like(real_pred)
    all_pred = torch.cat((real_pred, fake_pred))
    print("NON PATCH PRED", all_pred)
    all_y = torch.cat((y_real, y_fake))
    print("NON PATCH TARGETS", all_y)
    fpr, tpr, thresholds = roc_curve(all_y.cpu().numpy(), all_pred.cpu().numpy())
    pred_auc = auc(fpr, tpr)
    return pred_auc

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def validate_single_epoch(r_net, d_net, r_loss, d_loss, valid_loader, device, epoch) -> dict:
    r_net.eval()
    d_net.eval()

    valid_metrics = {'rec_loss': 0, 'gen_loss': 0, 'dis_loss': 0, 'auc': 0, 'patch_auc': 0}

    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):

            x_real = data[0].to(device)
            x_fake = r_net(x_real)

            # print("Real Prediction", torch.sigmoid(d_net(x_real)))
            # print("Fake Prediction", torch.sigmoid(d_net(x_fake)))

            dis_loss = d_loss(d_net, x_real, x_fake, do_print=False)

            r_metrics = r_loss(d_net, x_real, x_fake, 0)

            valid_metrics['rec_loss'] += r_metrics['rec_loss']
            valid_metrics['gen_loss'] += r_metrics['gen_loss']
            valid_metrics['dis_loss'] += dis_loss
            valid_metrics['auc'] += auc_pred(d_net, x_real, x_fake)
            valid_metrics['patch_auc'] += auc_patch_pred(d_net, x_real, x_fake)

            if epoch % 10 == 0:
                print(f'Saving test images on epoch {epoch}')
                # plot_learning_curves(metrics, metric_path)
                save_image(make_grid(x_real, nrows=10),
                           "../cifar_imgs/ae_recons/test_input_epoch_" + str(
                               epoch) + "_" + str(batch_idx) + ".jpg")
                save_image(make_grid(x_fake, nrows=10),
                           "../cifar_imgs/ae_recons/test_recon_epoch_" + str(
                               epoch) + "_" + str(batch_idx) + ".jpg")

    valid_metrics['rec_loss'] = valid_metrics['rec_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
    valid_metrics['gen_loss'] = valid_metrics['gen_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
    valid_metrics['dis_loss'] = valid_metrics['dis_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
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
