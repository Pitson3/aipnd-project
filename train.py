# Imports here
from argparse import ArgumentParser
import helper
import utility
import os
import torch
from torch import optim
import time
import numpy as np
from torchvision.models import resnet18, alexnet, vgg16, vgg11, vgg13

pre_trained_models = {'resnet18': resnet18, 'alexnet': alexnet,
                      'vgg16': vgg16, 'vgg11': vgg11, 'vgg13': vgg13}


def main():
    # load some attriputes from args
    in_args = load_args()
    # Get dataloaders, image_datasets
    dataloaders, image_datasets = utility.get_train_valid_test_loader(
        in_args.dir, in_args.gpu)

    # Build The model or rebuild it from checkpoint
    if in_args.new_model == False and os.path.isfile(in_args.save_dir):
        print('Rebuild the model from Existing checkpoint')
        print(f'checkpoint file: {in_args.save_dir}')
        model, optimizer, criterion, hidden_sizes = utility.load_checkpoint(
            in_args.save_dir)
    else:
        print(f'Create New Model based on pre_trained model {in_args.arch}')
        # choose arch pre_trained model
        pre_trained_model = pre_trained_models[in_args.arch](pretrained=True)
        hidden_sizes = in_args.hidden_units
        output_size = len(image_datasets['train'].classes)
        model = helper.get_classifier(
            pre_trained_model, hidden_sizes, output_size, dropHidden_p=0.5)
        optimizer = optim.Adam(model.classifier.parameters(),
                               lr=in_args.learning_rate)
        criterion = torch.nn.NLLLoss()

    # train
    measurements = helper.train(model, optimizer, criterion,
                                dataloaders['train'], dataloaders['valid'], in_args.gpu, epochs=in_args.epochs)
    # save checkpoint
    print(f"save checkpoint to {in_args.save_dir}")
    utility.save_checkpoint(in_args.save_dir, model, hidden_sizes,
                            image_datasets['train'], optimizer, criterion, measurements, in_args.arch, in_args.epochs)


def load_args():

    parser = ArgumentParser(description='Train the model')

    parser.add_argument('dir', type=str,
                        help='Path of the flowers directory to train the model')

    parser.add_argument('--new_model', action='store_true', default=False,
                        dest='new_model', help='Use to build new model, set a switch to true')
    parser.add_argument('--save_dir', type=str, default='./flower_checkpoint.pth',
                        help='Path checkpoint file to load/save')
    parser.add_argument('--arch', type=str, default='vgg11',
                        help='Type of pretrained model')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate hyperparameter')
    parser.add_argument('--hidden_units', type=int, default=[512], nargs='*',
                        help='hidden_units sequence')
    parser.add_argument('--epochs', type=int, default=1,
                        help="epochs hyperparameter")
    parser.add_argument('--gpu', action='store_true',
                        default=False,
                        dest='gpu',
                        help='Use GPU for inference, set a switch to true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
