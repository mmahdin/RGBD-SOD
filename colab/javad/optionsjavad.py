import argparse
import sys

version = 's1_x_x1_cross'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10,
                    help='training batch size')
parser.add_argument('--trainsize', type=int, default=352,
                    help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1,
                    help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='every n epochs decay learning rate')
parser.add_argument('--load', type=str,
                    default=f'/content/drive/My Drive/sod/BBS-Net/{version}/checkpoint.pth', help='train from checkpoints')
# parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

# ✅ Paths updated for Colab (Google Drive mount)
parser.add_argument('--rgb_root', type=str,
                    default='/content/dataset/RGBD_for_train/RGB/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str,
                    default='/content/dataset/RGBD_for_train/depth/', help='the training depth images root')
parser.add_argument('--gt_root', type=str,
                    default='/content/dataset/RGBD_for_train/GT/', help='the training gt images root')

parser.add_argument('--test_rgb_root', type=str,
                    default='/content/dataset/test_in_train/RGB/', help='the test rgb images root')
parser.add_argument('--test_depth_root', type=str,
                    default='/content/dataset/test_in_train/depth/', help='the test depth images root')
parser.add_argument('--test_gt_root', type=str,
                    default='/content/dataset/test_in_train/GT/', help='the test gt images root')

parser.add_argument('--save_path', type=str, default=f'/content/drive/My Drive/sod/BBS-Net/{version}/',
                    help='the path to save models and logs')

# ✅ Fix for Jupyter/Colab:
if 'ipykernel' in sys.modules:
    opt = parser.parse_args([])  # Ignore unknown args in notebook
else:
    opt = parser.parse_args()
