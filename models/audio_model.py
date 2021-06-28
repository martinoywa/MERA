# This file contains the audio model initializer
import torch.nn as nn


class AudioNN(nn.Module):
    def __init__(self):
        super(AudioNN, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(6),
        )

        self.seq2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(6),
        )

        self.dense1 = nn.Sequential(
            nn.Dropout(p=.5),
            nn.Linear(16 * 200 * 200, 2048),
        )

        self.dense2 = nn.Sequential(
            nn.Dropout(p=.5),
            nn.Linear(2048, 4)
        )

    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = Flatten.forward(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


class Flatten(nn.Module):
    def forward(self, x):
        # print(x.shape)
        return x.view(x.size(0), -1)
