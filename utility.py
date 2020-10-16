import helper
from PIL import Image
from torchvision import datasets, transforms, models
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import time
import json


def get_train_valid_test_loader(data_dir, gpu):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # some preperation variables
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = transforms.Resize(256)
    crop = transforms.CenterCrop(224)

    # train transform
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(
                                              45), transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(), normalize])
    # valid and test transform
    test_valid_transform = transforms.Compose(
        [resize, crop, transforms.ToTensor(), normalize])

    # TODO: Load the datasets with ImageFolder
    trainsets = datasets.ImageFolder(train_dir, transform=train_transform)
    validsets = datasets.ImageFolder(valid_dir, transform=test_valid_transform)
    testsets = datasets.ImageFolder(test_dir, transform=test_valid_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    cond_cuda = True if torch.cuda.is_available() and gpu else False
    if cond_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'loading data using device:{device}')
    trainloaders = torch.utils.data.DataLoader(
        trainsets, batch_size=32, shuffle=True, pin_memory=cond_cuda)
    validloaders = torch.utils.data.DataLoader(
        validsets, batch_size=64, shuffle=True, pin_memory=cond_cuda)
    testloaders = torch.utils.data.DataLoader(
        testsets, batch_size=64, shuffle=True, pin_memory=cond_cuda)

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {'train': train_transform,
                       'valid': test_valid_transform, 'test': test_valid_transform}

    # TODO: Load the datasets with ImageFolder
    image_datasets = {'train': trainsets, 'valid': validsets, 'test': testsets}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train': trainloaders,
                   'valid': validloaders, 'test': testloaders}

    return dataloaders, image_datasets


def load_checkpoint(path_checkpoint, train=True):
    checkpoint = torch.load(path_checkpoint)
    if checkpoint['Arch'] == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif checkpoint['Arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['Arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif checkpoint['Arch'] == 'resnet18':
        model = models.resnet18(pretrained=True)

    else:
        print('needs this pretrained model ....', checkpoint['Arch'])
        return
    hidden_sizes = checkpoint['hidden_sizes']
    output_size = checkpoint['output_size']
    dropHidden_p = checkpoint['dropHidden_p']
    dropIn_p = checkpoint['dropIn_p']
    classifier_state_dict = checkpoint['classifier_state_dict']
    helper.get_classifier(model, hidden_sizes,
                          output_size, dropIn_p, dropHidden_p)
    model.classifier.load_state_dict(classifier_state_dict)
    model.class_to_idx = checkpoint['class_to_idx']
    if not train:
        return model
    lr = checkpoint['learning_rate']
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = checkpoint['criterion']

    return model, optimizer, criterion, hidden_sizes


def save_checkpoint(path_checkpoint, model, hidden_sizes, trainsets, optimizer, criterion, measurements, arch, epochs, batch_size=32):
    model.class_to_idx = trainsets.class_to_idx

    checkpoint = {'batch_size': batch_size,
                  'Arch': arch,
                  'optimizer': optimizer.state_dict(),
                  'criterion': criterion,
                  'dropHidden_p': model.classifier.dropoutHidden.p,
                  'dropIn_p': model.classifier.dropoutIn.p,
                  'epoch': epochs,
                  'measurements': measurements,
                  'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
                  'input_size': model.classifier.state_dict()['hidden_layers.0.weight'].shape[1],
                  'output_size': len(trainsets.classes),
                  'hidden_sizes': hidden_sizes,
                  'classifier_state_dict': model.classifier.state_dict(),
                  'class_to_idx': model.class_to_idx
                  }
    # checkpoint;
    torch.save(checkpoint, path_checkpoint)


def CenterCrop(image, new_width=224):
    new_height = new_width
    width, height = image.size   # Get dimensions
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    return image


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    # Scales
    image.thumbnail((256, 256))
    # crops
    image = CenterCrop(image, 224)
    # convert into numpy array
    np_image = np.array(image)
    # normalizes in range [0,1]
    np_image = np_image / (np_image.max() - np_image.min())
    # normalizes image to match  the network
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for i in range(3):
        np_image[:, :, i] = (np_image[:, :, i] - mean[i])/std[i]
    # The color channel needs to be first
    # and retain the order of the other two dimensions.
    # np_image = np_image.transpose(2,0,1)

    # Add dim match model forward
    np_image = np.expand_dims(np_image, 1)
    # The color channel needs to be first
    # and retain the order of the other two dimensions.
    np_image = np_image.transpose(1, 3, 0, 2)
    torch_image = torch.from_numpy(np_image).float()
    return torch_image


def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file

    model.eval()
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() and gpu else 'cpu')
    print(f'prediction using device:{device}')
    model.to(device)
    start = time.time()
    # get PIL image
    pil_image = Image.open(image_path)
    # process image and return tensor torch image
    image = process_image(pil_image)
    image = image.to(device)
    # make forward pass
    with torch.no_grad():
        output = model.forward(image)
    # get probabilities
    ps = torch.exp(output)
    ps_topk = ps.topk(topk)
    # get topk probabilities
    probs = ps_topk[0].squeeze().tolist()
    # get topk indices of classes
    indices_class = ps_topk[1].squeeze().tolist()
    # invert the dictionary so you get a mapping from index to class
    my_inverted_dict = dict(map(reversed, model.class_to_idx.items()))
    # get topk of classes
    classes = [my_inverted_dict[i] for i in indices_class]
    # time_elapsed of prediction
    time_elapsed = (time.time() - start) * 1000
    print("\nTotal time: {:.0f}s {:.0f}ms".format(
        time_elapsed//1000, time_elapsed % 1000))
    return probs, classes


def sanityChecking(image_path, cat_to_name_json_path, model, gpu, topk=5):

    with open(cat_to_name_json_path, 'r') as f:
        cat_to_name_json = json.load(f)
    actual_class = image_path.split('/')[2]
    actual_class_name = cat_to_name_json[actual_class]
    probs, classes = predict(image_path, model, gpu, topk=5)

    #  convert from the class integer encoding to actual flower names
    classes_name = [cat_to_name_json[cl] for cl in classes]
    prediction_class_name = classes_name[0]

    print(f'Accuracy {prediction_class_name == actual_class_name}')
    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot2grid((15, 9), (0, 0), colspan=9, rowspan=9)
    ax2 = plt.subplot2grid((15, 9), (10, 2), colspan=5, rowspan=5)

    image = Image.open(image_path)
    # draw predicted image
    ax1.imshow(image)
    ax1.set_title(prediction_class_name)
    # Build bar probs
    ax2.barh(classes_name, probs, color='red')
    ax2.set_xlabel('Probabilities')
    ax2.invert_yaxis()
    plt.show()
    return classes_name, probs
