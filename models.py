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

        # 1st layer convolution network; input (1, 224, 224), output: (32, 224, 224)
        # 1 input image channel (grayscale), 32 output channels/feature maps, 3x3 square convolution kernel
        # stride = 1, same padding, dilation =1
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=0, dilation=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        # max pooling layer output: (32, 112, 112)
        self.pooling1 = nn.MaxPool2d(2)

        # 2nd layer convolution network; input (32, 112, 112), output: (64, 112, 112)
        # 32 input image channel, 64 output channels/feature maps, 3x3 square convolution kernel
        # stride = 1, same padding, dilation =1
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=0, dilation=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        # max pooling layer output: (64, 56, 56)
        self.pooling2 = nn.MaxPool2d(2)

        # 3rd layer convolution network; input (1, 56, 56), output: (64, 56, 56)
        # 64 input image channel, 128 output channels/feature maps, 3x3 square convolution kernel
        # stride = 1, same padding, dilation =1
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=0, dilation=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        # max pooling layer output: (128, 28, 28)
        self.pooling3 = nn.MaxPool2d(2)


        # 4th layer convolution network; input (128, 56, 56), output: (256, 56, 56)
        # 128 input image channel, 256 output channels/feature maps, 3x3 square convolution kernel
        # stride = 1, same padding, dilation =1
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=0, dilation=1)
        self.batch_norm4 = nn.BatchNorm2d(256)
        # max pooling layer output: (256, 14, 14)
        self.pooling4 = nn.MaxPool2d(2)
        
        # flatten: input (128, 56, 56), output (50176)
        # 5th layer fully connected: input 50176, output: 24000
        self.fc1 = nn.Linear(36864, 24000)
        # dropout; keep_probability = 0.5
        self.dropout1 = nn.Dropout(p=0.5)
        
        # 6th layer fully connected: input 24000, output: 10000
        self.fc2 = nn.Linear(24000, 10000)
        # dropout; keep_probability = 0.5
        self.dropout2 = nn.Dropout(p=0.5)

        # 7th layer fully connected: input 10000, output: 4000
        self.fc3 = nn.Linear(10000, 4000)
        # dropout; keep_probability = 0.5
        self.dropout3 = nn.Dropout(p=0.5)

        # 8th layer fully connected: input 4000, output: 1000
        self.fc4 = nn.Linear(4000, 1000)
        # dropout; keep_probability = 0.5
        self.dropout4 = nn.Dropout(p=0.5)

        # 9th layer fully connected: input 1000, output: 400
        self.fc5 = nn.Linear(1000, 400)
        # dropout; keep_probability = 0.5
        self.dropout5 = nn.Dropout(p=0.5)

        # 10th layer fully connected: input 400, output: 136
        self.fc6 = nn.Linear(400, 136)
        # dropout; keep_probability = 0.5
        # self.dropout5 = nn.Dropout(p=0.5)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # Convolution
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pooling1(x)
        
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pooling2(x)
        
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.pooling3(x)
        
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.pooling4(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
       
        x = self.fc5(x)
        x = self.dropout5(x)
        # a modified x, having gone through all the layers of your model, should be returned
        x = self.fc6(x)
        return x
