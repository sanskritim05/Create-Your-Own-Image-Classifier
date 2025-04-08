import argparse
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils import data
import os

def data_transforms_custom(args):
    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")

    for directory in [args.data_dir, args.save_dir, train_dir, valid_dir]:
        if not os.path.exists(directory):
            print(f"Directory doesn't exist: {directory}")
            raise FileNotFoundError

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = ImageFolder(root=train_dir, transform=train_transforms)
    valid_data = ImageFolder(root=valid_dir, transform=valid_transforms)

    train_data_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_data_loader = data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)

    return train_data_loader, valid_data_loader, train_data.class_to_idx


def train_custom_model(args, train_data_loader, valid_data_loader, class_to_idx):
    model = torchvision.models.__dict__[args.architecture](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(2048, args.num_classes)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model.to(device)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_data_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % args.print_every == args.print_every - 1:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / args.print_every))
                running_loss = 0.0

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valid_data_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the %d validation images: %d %%' % (total, 100 * correct / total))

    model.class_to_idx = class_to_idx
    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'args': args
                  }

    torch.save(checkpoint, os.path.join(args.save_dir, "checkpoint.pth"))
    print("Model saved to {}".format(os.path.join(args.save_dir, "checkpoint.pth")))
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(dest='data_dir',
                        help="Directory containing the training images. Should have 'train' and 'valid' subdirectories.")
    parser.add_argument('--save_dir', dest='save_dir',
                        help="Directory where the trained model will be saved.", default='./saved_models')
    parser.add_argument('--learning_rate', dest='learning_rate',
                        help="Learning rate for training the model.", default=0.001, type=float)
    parser.add_argument('--epochs', dest='epochs',
                        help="Number of epochs for training.", default=10, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        help="Batch size for training.", default=64, type=int)
    parser.add_argument('--gpu', dest='gpu',
                        help="Use GPU for training if available.", action='store_true')
    parser.add_argument('--architecture', dest='architecture',
                        help="Pre-trained model architecture.", default="resnet50", type=str)
    parser.add_argument('--num_classes', dest='num_classes',
                        help="Number of classes in the dataset.", default=102, type=int)
    parser.add_argument('--print_every', dest='print_every',
                        help="Print loss every specified number of steps.", default=20, type=int)

    args = parser.parse_args()

    train_data_loader, valid_data_loader, class_to_idx = data_transforms_custom(args)
    train_custom_model(args, train_data_loader, valid_data_loader, class_to_idx)
