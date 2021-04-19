import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim

from audio_model import AudioNN, Flatten
from data_loaders import data


# model = AudioNN()
model = nn.Sequential(
    nn.Conv2d(3, 8, 5),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(8, 16, 5),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, 5),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, 5),
    nn.MaxPool2d(2, 2),
    Flatten(),
    nn.Linear(64 * 46 * 71, 1000),
    nn.ReLU(),
    nn.Linear(1000, 4)
)
# if cuda.is_available():
#     model.cuda()

trainloader, testloader = data('C:\\Users\\marti\\PycharmProjects\\MERA\\data\\train\\spectrograms')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(number_epochs, model_version):
    # prints out model training progress
    for epoch in range(number_epochs):
        train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            inputs, labels = data
            # if cuda.is_available():
            #     inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.shape, labels.shape)
            # print(outputs)
            # print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*inputs.size(0)

        # print training statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(trainloader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))

    print('Finished Training')
    torch.save(model.state_dict(), 'model'+model_version)


def test(model_weights):
    # uses the saved model weights to test trained
    # model performance on test set and returns the
    # results
    model.load_state_dict(torch.load(model_weights, map_location='cpu'), strict=False)
    model.eval()

    dataiter = iter(trainloader)
    inputs, labels = dataiter.next()

    outputs = model(inputs)
    _, preds = torch.max(outputs, dim=1)

    print(f'{preds}\n'
          f'{labels}')


if __name__ == '__main__':
    number_epochs = 30
    model_version = '-0.0.2'
    train(number_epochs, model_version)

    model_weights = "model" + model_version
    test(model_weights)
