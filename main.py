import os
import argparse
import warnings
warnings.simplefilter('ignore')
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)


    data_loader = get_loader(config.crop_size, config.image_size, config.batch_size,
                            config.dataset, config.mode, config.num_workers, config.line_type)
    
    solver = Solver(data_loader, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--crop_size', type=int, default=256, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=276, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=16, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--d_channel', type=int, default=448)
    parser.add_argument('--channel_1x1', type=int, default=256)
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_rec', type=float, default=30, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_perc', type=float, default=0.01)
    parser.add_argument('--lambda_style', type=float, default=50)
    parser.add_argument('--lambda_tr', type=float, default=1)
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='line_art') # , choices=['line_art, tag2pix']
    parser.add_argument('--line_type', type=str, default='xdog') # , choices=['xdog, keras']
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_epoch', type=int, default=200, help='number of total iterations for training D')
    parser.add_argument('--num_epoch_decay', type=int, default=100, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')

    # Test configuration.
    parser.add_argument('--test_epoch', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Directories.
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--exp_name', type=str, default='baseline')

    # Step size.
    parser.add_argument('--log_step', type=int, default=200)
    parser.add_argument('--sample_epoch', type=int, default=1)
    parser.add_argument('--model_save_step', type=int, default=40)

    config = parser.parse_args()
    config.log_dir = os.path.join(config.result_dir, config.exp_name, 'log')
    config.sample_dir = os.path.join(config.result_dir, config.exp_name, config.exp_name)
    config.model_save_dir = os.path.join(config.result_dir, config.exp_name, 'model')
    print(config)
    main(config)