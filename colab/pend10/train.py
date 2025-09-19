from data import get_loader, test_dataset
from pend10.options import opt
import torch.backends.cudnn as cudnn
import logging
from tensorboardX import SummaryWriter
from pend10.model import BBSNetTransformerAttention as BBSNet
# from models.BBSNet_model import BBSNetSwin as BBSNet
from torchvision.utils import make_grid
from datetime import datetime
import numpy as np
import os
import torch
import torch.nn.functional as F
import sys
import csv

sys.path.append('./models')

# =======================
# Device setup
# =======================
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

# =======================
# Model & optimizer
# =======================
model = BBSNet()
optimizer = torch.optim.Adam(model.parameters(), opt.lr)

start_epoch = 1
best_mae = 1
best_epoch = 0
# Store individual dataset MAE scores for the best model
best_dataset_maes = {}

# =======================
# Load checkpoint helper
# =======================


def load_optimizer_state_to_cuda(optimizer, checkpoint_state, device='cuda'):
    optimizer.load_state_dict(checkpoint_state)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def load_model_with_lazy_unembed(model, state_dict):
    """
    Load model partially (strict=False).
    Keep _out_proj weights aside until those layers are built.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    pending_unembed = {k: v for k,
                       v in state_dict.items() if "._out_proj." in k}
    return pending_unembed


def restore_unembed_weights(model, pending_unembed):
    """
    Copy stored _out_proj weights into the model after they exist.
    """
    own_state = model.state_dict()
    for k, v in pending_unembed.items():
        if k in own_state:
            own_state[k].copy_(v)


# =======================
# Load checkpoint if provided
# =======================
pending_unembed = None
if opt.load is not None and os.path.exists(opt.load):
    print(opt.load)
    checkpoint = torch.load(opt.load, weights_only=False)
    if isinstance(checkpoint, dict):  # Resuming from full checkpoint
        pending_unembed = load_model_with_lazy_unembed(
            model, checkpoint['model_state'])
        load_optimizer_state_to_cuda(
            optimizer, checkpoint['optimizer_state'], device='cuda')
        start_epoch = checkpoint['epoch'] + 1
        best_mae = checkpoint.get('best_mae', 1)
        best_epoch = checkpoint.get('best_epoch', 0)
        best_dataset_maes = checkpoint.get('best_dataset_maes', {})
        print(f"Resumed from epoch {checkpoint['epoch']}, best MAE={best_mae}")
    else:  # Loading only model weights
        pending_unembed = load_model_with_lazy_unembed(model, checkpoint)
        print(f"Loaded weights from {opt.load}")

model.cuda()

# =======================
# Paths
# =======================
save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Define test datasets
test_datasets = ['NJU2K', 'NLPR', 'STERE', 'DES', 'SSD', 'LFSD', 'SIP']

# Loss CSV file - updated header to include individual dataset MAE scores
loss_log_file = os.path.join(save_path, 'loss_log.csv')
if not os.path.exists(loss_log_file):
    with open(loss_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['epoch', 'train_loss', 'average_test_mae'] + \
            [f'{dataset}_mae' for dataset in test_datasets]
        writer.writerow(header)

# =======================
# Data loaders
# =======================
print('Loading data...')
train_loader = get_loader(opt.rgb_root, opt.gt_root, opt.depth_root,
                          batchsize=opt.batchsize, trainsize=opt.trainsize)

# Create test loaders for each dataset
test_loaders = {}
for dataset in test_datasets:
    rgb_path = os.path.join(opt.test_rgb_root, dataset, 'RGB') + '/'
    gt_path = os.path.join(opt.test_gt_root, dataset, 'GT') + '/'
    depth_path = os.path.join(opt.test_depth_root, dataset, 'depth') + '/'

    # Check if dataset exists
    if os.path.exists(rgb_path) and os.path.exists(gt_path) and os.path.exists(depth_path):
        test_loaders[dataset] = test_dataset(
            rgb_path, gt_path, depth_path, opt.trainsize)
        print(f"Loaded test dataset: {dataset}")
    else:
        print(f"Warning: Dataset {dataset} not found at expected paths")
        print(
            f"Expected paths: RGB={rgb_path}, GT={gt_path}, depth={depth_path}")

total_step = len(train_loader)

# =======================
# Logging
# =======================
logging.basicConfig(filename=os.path.join(save_path, 'log.log'),
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("BBSNet-Train Resume")
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
    opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path, opt.decay_epoch))

# Loss function
CE = torch.nn.BCEWithLogitsLoss()

# TensorBoard
writer = SummaryWriter(save_path + 'summary')
step = 0


def structure_loss(pred, mask):
    weit = 1 + 5 * \
        torch.abs(F.avg_pool2d(mask, kernel_size=31,
                  stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path, pending_unembed=None):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0

    for i, (images, gts, depths) in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        images = images.cuda()
        gts = gts.cuda()
        depths = depths.cuda()

        s1, s2 = model(images, depths)

        # 🔥 After the first forward pass, _out_proj layers are created → restore weights if any
        if pending_unembed is not None:
            restore_unembed_weights(model, pending_unembed)
            print("Restored _out_proj weights into model.")
            pending_unembed = None  # restore only once

        loss1 = structure_loss(s1, gts)
        loss2 = structure_loss(s2, gts)
        loss = loss1 + loss2
        # loss1 = CE(s1, gts)
        # loss2 = CE(s2, gts)
        # loss = loss1 + loss2
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        step += 1
        epoch_step += 1
        loss_all += loss.data

        if i % 100 == 0 or i == total_step or i == 1:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.format(
                datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data))
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.format(
                epoch, opt.epoch, i, total_step, loss1.data, loss2.data))
            writer.add_scalar('Loss', loss.data, global_step=step)

            grid_image = make_grid(
                images[0].clone().cpu().data, 1, normalize=True)
            writer.add_image('RGB', grid_image, step)
            grid_image = make_grid(
                gts[0].clone().cpu().data, 1, normalize=True)
            writer.add_image('Ground_truth', grid_image, step)

            for idx, out in enumerate([s1, s2], 1):
                res = out[0].clone().sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image(f's{idx}', torch.tensor(
                    res), step, dataformats='HW')

    loss_all /= epoch_step
    logging.info(
        '#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
    writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

    return loss_all


def test(test_loaders, model, epoch, save_path):
    global best_mae, best_epoch, best_dataset_maes
    model.eval()

    dataset_maes = {}
    total_mae_sum = 0
    total_datasets = 0

    with torch.no_grad():
        for dataset_name, test_loader in test_loaders.items():
            mae_sum = 0
            for i in range(test_loader.size):
                image, gt, depth, name, img_for_post = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                depth = depth.cuda()
                _, res = model(image, depth)
                res = F.interpolate(res, size=gt.shape,
                                    mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                mae_sum += np.sum(np.abs(res - gt)) / \
                    (gt.shape[0] * gt.shape[1])

            dataset_mae = mae_sum / test_loader.size
            dataset_maes[dataset_name] = dataset_mae
            total_mae_sum += dataset_mae
            total_datasets += 1

            # Log individual dataset MAE to tensorboard
            writer.add_scalar(f'MAE_{dataset_name}', torch.tensor(
                dataset_mae), global_step=epoch)
            print(f"Epoch: {epoch} {dataset_name} MAE: {dataset_mae:.4f}")

    # Calculate average MAE across all datasets
    average_mae = total_mae_sum / \
        total_datasets if total_datasets > 0 else float('inf')
    writer.add_scalar('MAE_Average', torch.tensor(
        average_mae), global_step=epoch)

    print(
        f"Epoch: {epoch} Average MAE: {average_mae:.4f} ####  bestMAE: {best_mae:.4f} bestEpoch: {best_epoch}")

    # Save test MAE to CSV
    with open(loss_log_file, 'a', newline='') as f:
        writer_csv = csv.writer(f)
        row = [epoch, '', float(average_mae)]
        # Add individual dataset MAE scores
        for dataset in test_datasets:
            if dataset in dataset_maes:
                row.append(float(dataset_maes[dataset]))
            else:
                row.append('')  # Empty if dataset not available
        writer_csv.writerow(row)

    # Check if this is the best model (lower average MAE is better)
    if epoch == 1:
        best_mae = average_mae
        best_dataset_maes = dataset_maes.copy()
    else:
        if average_mae < best_mae:
            best_mae = average_mae
            best_epoch = epoch
            best_dataset_maes = dataset_maes.copy()
            torch.save(model.state_dict(), os.path.join(
                save_path, 'BBSNet_epoch_best.pth'))
            print(
                f"Best epoch updated: {epoch} with average MAE: {average_mae:.4f}")

            # Log individual MAE scores for the best model
            mae_details = " | ".join(
                [f"{dataset}: {mae:.4f}" for dataset, mae in dataset_maes.items()])
            print(f"Individual MAE scores: {mae_details}")

    logging.info('#TEST#:Epoch:{} AverageMAE:{:.4f} bestEpoch:{} bestMAE:{:.4f}'.format(
        epoch, average_mae, best_epoch, best_mae))

    # Log individual dataset results
    for dataset_name, mae in dataset_maes.items():
        logging.info('#TEST#:Epoch:{} {}_MAE:{:.4f}'.format(
            epoch, dataset_name, mae))


def main():
    print("Start training...")
    for epoch in range(start_epoch, opt.epoch + 1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch,
                           opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

        train_loss = train(train_loader, model, optimizer, epoch, save_path)
        test(test_loaders, model, epoch, save_path)

        # Save checkpoint with additional information
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_mae': best_mae,
            'best_epoch': best_epoch,
            'best_dataset_maes': best_dataset_maes
        }, os.path.join(save_path, 'checkpoint.pth'))

        # Save train loss to CSV (separate row for training info)
        with open(loss_log_file, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            row = [epoch, float(train_loss)] + [''] * (len(test_datasets) + 1)
            writer_csv.writerow(row)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * (0.1**(epoch//40))
        lr = param_group['lr']
    return lr


if __name__ == '__main__':
    main()
