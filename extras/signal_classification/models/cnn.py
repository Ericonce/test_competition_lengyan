import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size, stride),
            nn.BatchNorm1d(out_planes),
            nn.ReLU()
            # nn.LeakyReLU()
        )


class CNN(nn.Module):
    def __init__(self, num_classes, in_channel):
        super(CNN, self).__init__()
        self.conv1 = ConvBNReLU(in_channel, 16, 3)
        self.conv2 = ConvBNReLU(16, 16, 3)
        self.pool1 = nn.MaxPool1d(2)

        self.conv3 = ConvBNReLU(16, 32, 5)
        self.conv4 = ConvBNReLU(32, 32, 5)
        self.pool2 = nn.MaxPool1d(2)

        self.conv5 = ConvBNReLU(32, 64, 7)
        self.conv6 = ConvBNReLU(64, 64, 7)
        self.pool3 = nn.MaxPool1d(2)

        self.conv7 = ConvBNReLU(64, 128, 9)
        self.conv8 = ConvBNReLU(128, 128, 9)
        self.pool4 = nn.MaxPool1d(2)

        self.conv9 = ConvBNReLU(128, 256, 11)
        self.conv10 = ConvBNReLU(256, 256, 11)
        self.pool5 = nn.MaxPool1d(2)

        self.fc1 = nn.Sequential(
            nn.Linear(19712, 256),
            nn.Dropout(0.2))
        self.fc2 = nn.Sequential(
            nn.Linear(256, num_classes))

    def forward(self, out):
        out = self.conv1(out)
        out = self.pool1(self.conv2(out))
        out = self.conv3(out)
        out = self.pool2(self.conv4(out))
        out = self.conv5(out)
        out = self.pool3(self.conv6(out))
        out = self.conv7(out)
        out = self.pool4(self.conv8(out))
        out = self.conv9(out)
        out = self.pool5(self.conv10(out))
        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)

        return out

    def feature(self, out):
        out = self.conv1(out)
        out = self.pool1(self.conv2(out))
        out = self.conv3(out)
        out = self.pool2(self.conv4(out))
        out = self.conv5(out)
        out = self.pool3(self.conv6(out))
        out = self.conv7(out)
        out = self.pool4(self.conv8(out))
        out = self.conv9(out)
        out = self.pool5(self.conv10(out))
        feature = out.reshape(out.size(0), -1)

        return feature