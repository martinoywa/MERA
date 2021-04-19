import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim

from audio_model import AudioNN
from data_loaders import data


model = AudioNN()
# if cuda.is_available():
#     model.cuda()

trainloader, testloader = data('/home/martin/Projects/MERA/data/train/spectrograms')

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
            loss = criterion(labels, outputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, train_loss))
            train_loss = 0.0

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

    print(outputs)
    print(f'{preds, labels}')
