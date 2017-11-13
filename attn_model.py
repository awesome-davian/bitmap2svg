import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from Attention import Attn

# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet Module for Show Attend and Tell model 
class AttnEncoder(nn.Module):
    def __init__(self, block, layers):
        super(AttnEncoder, self).__init__()
        self.in_channels = 32
        self.conv = conv3x3(3, 32)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 32, layers[0])
        self.layer2 = self.make_layer(block, 64, layers[0],2)
        self.layer3 = self.make_layer(block, 128, layers[1],2)
        #self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)
        self.fc2.weight.data.normal_(0.0, 0.02)
        self.fc2.bias.data.fill_(0)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), out.size(1), -1)
        return out


class AttnDecoderRnn(nn.Module):
    def __init__(self,  feature_size, hidden_size, vocab_size, num_layers):
        super(AttnDecoderRnn, self).__init__()
        #Define parameters

        #Define layers
        self.embed = nn.Embedding(vocab_size, feature_size)
        self.init_layer = nn.Linear(feature_size, hidden_size)
        self.attn = Attn('general', feature_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.ctx2out = nn.Linear(feature_size, feature_size)
        self.h2out = nn.Linear(hidden_size, feature_size)
        self.out = nn.Linear(feature_size, vocab_size)
    
    def decode_lstm(self, input_word, context, hidden, lstm_out):

        #hidden = hidden.squeeze(0)
        #out = self.h2out(hidden)
        lstm_out = lstm_out.squeeze(1)
        out = self.h2out(lstm_out)
        context = context.squeeze(1)
        out += self.ctx2out(context)
        out += input_word
       
        out = self.out(out)

        return out

    def init_lstm(self, features):

        sums = torch.sum(features, 1)
        out = torch.mul(sums, 1/features.size(1))
        out = out.squeeze(1).unsqueeze(0) # 1, batch, feature_size
        out = self.init_layer(out.squeeze(0)).unsqueeze(0)

        return out, out 


    def forward(self, features, captions, lengths):

        max_length = max(lengths)
        embed = self.embed(captions)
        h, c= self.init_lstm(features)
        arr = [] 

        for i in range(max_length):
            context = self.attn(h, features)
            if i == 0 :
                input_word = Variable(torch.zeros(embed.size(0), embed.size(2))).cuda()
            else: 
                input_word = embed[:,i-1]
            lstm_input = torch.cat((context, input_word.unsqueeze(1)),2)
            lstm_out, (h,c) = self.lstm(lstm_input, (h,c))
            out = self.decode_lstm(input_word,context, h, lstm_out).unsqueeze(1)

            arr += [out]

        return torch.cat(arr,1)

    def sample(self, features):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        h,c = self.init_lstm(features)

        for i in range(30):                                      # maximum sampling length
            if i == 0:
                word_init = Variable(torch.LongTensor([1])).cuda()
                #x = self.embed(word_init).unsqueeze(1)
                x = Variable(torch.rand(1,1,256)).cuda()
                #sampled_ids.append(word_init)
            else: 
                predicted = predicted
                x = self.embed((predicted))

            context = self.attn(h, features)
            lstm_input = torch.cat((context, x) ,2)
            lstm_out, (h,c) = self.lstm(lstm_input, (h,c))          # (batch_size, 1, hidden_size), 
            out = self.decode_lstm(x, context, h, lstm_out)
            print(out)
            predicted = out.max(1)[1]
            print(predicted)
            sampled_ids.append(predicted)
        #print(sampled_ids)
        #sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids













