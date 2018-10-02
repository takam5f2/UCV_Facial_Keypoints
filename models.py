## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        # input image size(channel, height, width) = (1, 224, 224)
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        # 1st layer convolution network; input (1, 224, 224), output: (4, 224, 224)
        # stride = 1, same padding, dilation =1
        self.conv1 = nn.Conv2d(1, 4, 3, stride=1, padding=1, dilation=1)
        self.batch_norm1 = nn.BatchNorm2d(4)
        # 2nd layer convolution network; input (4, 224, 224), output: (4, 224, 224)
        # stride = 1, same padding, dilation =1
        self.conv2 = nn.Conv2d(4, 4, 3, stride=1, padding=1, dilation=1)
        self.batch_norm2 = nn.BatchNorm2d(4)
        # max pooling layer output: (4, 112, 112)
        self.pooling1 = nn.MaxPool2d(2)

        # 3rd layer convolution network; input (4, 112, 112), output: (8, 112, 112)
        # stride = 1, same padding, dilation =1
        self.conv3 = nn.Conv2d(4, 8, 3, stride=1, padding=1, dilation=1)
        self.batch_norm3 = nn.BatchNorm2d(8)
        # 4th layer convolution network; input (8, 112, 112), output: (8, 112, 112)
        # stride = 1, same padding, dilation =1
        self.conv4 = nn.Conv2d(8, 8, 3, stride=1, padding=1, dilation=1)
        self.batch_norm4 = nn.BatchNorm2d(8)
        # max pooling layer output: (8, 56, 56)
        self.pooling2 = nn.MaxPool2d(2)
        
        # 5th layer convolution network; input (8, 56, 56), output: (16, 56, 56)
        # stride = 1, same padding, dilation =1
        self.conv5 = nn.Conv2d(8, 16, 3, stride=1, padding=1, dilation=1)
        self.batch_norm5 = nn.BatchNorm2d(16)
        # 6th layer convolution network; input (8, 56, 56), output: (8, 56, 56)
        # stride = 1, same padding, dilation =1
        self.conv6 = nn.Conv2d(16, 16, 3, stride=1, padding=1, dilation=1)
        self.batch_norm6 = nn.BatchNorm2d(16)
        # max pooling layer output: (16, 56, 56)
        self.pooling3 = nn.MaxPool2d(2)

        # 7th layer convolution network; input (16, 56, 56), output: (32, 56, 56)
        # stride = 1, same padding, dilation =1
        self.conv7 = nn.Conv2d(16, 32, 3, stride=1, padding=1, dilation=1)
        self.batch_norm7 = nn.BatchNorm2d(32)
        # 8th layer convolution network; input (32, 56, 56), output: (32, 56, 56)
        # stride = 1, same padding, dilation =1
        self.conv8 = nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=1)
        self.batch_norm8 = nn.BatchNorm2d(32)
        # max pooling layer output: (32, 28, 28)
        self.pooling4 = nn.MaxPool2d(2)
        
        # 9th layer convolution network; input (16, 28, 28), output: (32, 28, 28)
        # stride = 1, same padding, dilation =1
        self.conv9 = nn.Conv2d(32, 64, 3, stride=1, padding=1, dilation=1)
        self.batch_norm9 = nn.BatchNorm2d(64)
        # 10th layer convolution network; input (32, 28, 28), output: (32, 28, 28)
        # stride = 1, same padding, dilation =1
        self.conv10 = nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1)
        self.batch_norm10 = nn.BatchNorm2d(64)
        # 11th layer convolution network; input (32, 28, 28), output: (32, 28, 28)
        # stride = 1, same padding, dilation =1
        self.conv11 = nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1)
        self.batch_norm11 = nn.BatchNorm2d(64)
        # max pooling layer output: (64, 14, 14)
        self.pooling5 = nn.MaxPool2d(2)

        # 12th layer convolution network; input (128, 14, 14), output: (128, 14, 14)
        # stride = 1, same padding, dilation =1
        self.conv12 = nn.Conv2d(64, 128, 3, stride=1, padding=1, dilation=1)
        self.batch_norm12 = nn.BatchNorm2d(128)
        # 13th layer convolution network; input (128, 14, 14), output: (128, 14, 14)
        # stride = 1, same padding, dilation =1
        self.conv13 = nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1)
        self.batch_norm13 = nn.BatchNorm2d(128)
        # 14th layer convolution network; input (128, 14, 14), output: (128, 14, 14)
        # stride = 1, same padding, dilation =1
        self.conv14 = nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1)
        self.batch_norm14 = nn.BatchNorm2d(128)
        # max pooling layer output: (128, 7, 7)
        self.pooling6 = nn.MaxPool2d(2)
        
        # 15th layer convolution network; input (128, 7, 7), output: (28, 7, 7)
        # stride = 1, same padding, dilation =1
        self.conv15 = nn.Conv2d(128, 256, 3, stride=1, padding=1, dilation=1)
        
        # Fully Connected
        self.fc1 = nn.Linear(2304, 1000)
        self.dropout1 = nn.Dropout(0.5)
        # Fully Connected
        self.fc2 = nn.Linear(1000, 1000)
        self.dropout2 = nn.Dropout(0.5)
        # Fully Connected
        self.fc3 = nn.Linear(1000, 136)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # Convolution
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pooling1(x)
        
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.pooling2(x)
        
        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = F.relu(self.batch_norm6(self.conv6(x)))
        x = self.pooling3(x)

        x = F.relu(self.batch_norm7(self.conv7(x)))
        x = F.relu(self.batch_norm8(self.conv8(x)))
        x = self.pooling4(x)
        
        x = F.relu(self.batch_norm9(self.conv9(x)))
        x = F.relu(self.batch_norm10(self.conv10(x)))
        x = F.relu(self.batch_norm11(self.conv11(x)))
        x = self.pooling5(x)

        x = F.relu(self.batch_norm12(self.conv12(x)))
        x = F.relu(self.batch_norm13(self.conv13(x)))
        x = F.relu(self.batch_norm14(self.conv14(x)))
        x = self.pooling6(x)

        x = F.relu(self.conv15(x))
        # x = F.relu(self.batch_norm3(self.conv3(x)))
        # x = self.pooling3(x)
        
        # x = F.relu(self.batch_norm4(self.conv4(x)))
        # x = self.pooling4(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        # x = self.dropout3(x)

        # x = F.relu(self.fc4(x))
        # x = self.dropout4(x)
        # x = self.fc4(x)
        # x = self.fc5(x)
        # x = self.dropout5(x)
        # a modified x, having gone through all the layers of your model, should be returned
        # x = self.fc6(x)
        return x
