import os
import argparse
from data_loader import get_loader
from solver import Solver
from torch.backends import cudnn


def main(config):

    cudnn.benchmark = True
    data_loader = get_loader(config.model, config.image_path, config.metadata_path, config.mode, config.batch_size,
                             config.shuffle)
    # print (config.model)
    # print (config.image_path)
    # print (config.metadata_path)
    # print (config.sum_mode)

    solver = Solver(data_loader, config)
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--sum_mode', type=bool, default=False)
    parser.add_argument('--bayesian', type=bool, default=True, help='Bayesian Posenet, True or False')
    parser.add_argument('--sequential_mode', type=str, default=None,
                        choices=[None, 'fixed_weight', 'batch_size', 'learning_rate', 'beta'])
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta', type=float, default=500)
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='range 0.0 to 1.0')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--fixed_weight', type=bool, default=False)
    parser.add_argument('--model', type=str, default='Resnet', choices=['Googlenet', 'Resnet'])
    parser.add_argument('--pretrained_model', type=str, default=None)

    parser.add_argument('--image_path', type=str, default='/mnt/data2/image_based_localization/posenet/KingsCollege')
    parser.add_argument('--metadata_path', type=str,
                        default='/mnt/data2/image_based_localization/posenet/KingsCollege/dataset_test.txt')

    # Training settings
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0 1 2 3')
    # parser.add_argument('--dataset', type=str, default='Oxford', choices=['NCLT', 'VKITTI', 'Oxford', 'QUT'])
    parser.add_argument('--num_epochs', type=int, default= 300)
    parser.add_argument('--num_epochs_decay', type=int, default=10)
    parser.add_argument('--num_iters', type=int, default= 300) # 200000
    parser.add_argument('--num_iters_decay', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default= 1) # 16
    parser.add_argument('--num_workers', type=int, default=1)

    # Test settings
    parser.add_argument('--test_model', type=str, default= None)
    parser.add_argument('--target_attributes', nargs='+', default=['overcast', 'night'])

    # Misc
    parser.add_argument('--use_tensorboard', type=bool, default=True)
    parser.add_argument('--save_result', type=bool, default=True)

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1000)

    config = parser.parse_args()
    main(config)
