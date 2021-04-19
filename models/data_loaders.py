import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader


def data(data_path):
    # returns train, testloaders
    transform = transforms.Compose([
        transforms.Resize(800),
        transforms.ToTensor(),
    ])

    data = datasets.ImageFolder(data_path, transform=transform)
    train_size = int(len(data)*0.8)
    test_size = len(data)-train_size
    trainset, testset = random_split(data, lengths=[train_size, test_size])

    trainloader = DataLoader(trainset, batch_size=16)
    testloader = DataLoader(testset)

    return trainloader, testloader
