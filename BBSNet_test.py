import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2

from models.BBSNet_model import BBSNetChannelSpatialAttention as BBSNet
# from models.BBSNet_model import BBSNetTransformerAttention as BBSNet
from data import test_dataset


def load_optimizer_state_to_cuda(optimizer, checkpoint_state, device="cuda"):
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


def test(method="patchify_light_pos_embed"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--testsize", type=int,
                        default=352, help="testing size")
    parser.add_argument("--gpu_id", type=str,
                        default="0", help="select gpu id")
    parser.add_argument(
        "--test_path",
        type=str,
        default="../BBS_dataset/RGBD_for_test/",
        help="test dataset path",
    )
    opt = parser.parse_args(args=[])

    dataset_path = opt.test_path

    # set device for test
    if opt.gpu_id == "0":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("USE GPU 0")
    elif opt.gpu_id == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print("USE GPU 1")

    # load the model
    model = BBSNet()
    checkpoint = torch.load(
        f"BBSNet_cpts/{method}/BBSNet_epoch_best.pth")

    # lazy load weights
    pending_unembed = load_model_with_lazy_unembed(model, checkpoint)

    model.cuda()
    model.eval()

    # test
    # test_datasets = ["test_in_train"]
    test_datasets = ['NJU2K', 'NLPR', 'STERE', 'DES', 'SSD', 'LFSD', 'SIP']
    pending_restored = False
    for dataset in test_datasets:
        save_path = f"./pred/{method}/{dataset}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + "/RGB/"
        gt_root = dataset_path + dataset + "/GT/"
        depth_root = dataset_path + dataset + "/depth/"
        test_loader = test_dataset(
            image_root, gt_root, depth_root, opt.testsize)

        for i in range(test_loader.size):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= gt.max() + 1e-8

            image = image.cuda()
            depth = depth.cuda()

            # first forward pass will build _out_proj
            _, res = model(image, depth)

            # restore unembed weights once
            if (not pending_restored) and (len(pending_unembed) > 0):
                restore_unembed_weights(model, pending_unembed)
                print("Restored _out_proj weights into model.")
                pending_restored = True

            res = F.upsample(res, size=gt.shape,
                             mode="bilinear", align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            print("save img to: ", save_path + name)
            cv2.imwrite(save_path + name, res * 255)
        print("Test Done!")
