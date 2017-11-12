import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from data_loader import build_vocab, get_loader
from model import EncoderCNN, DecoderRNN 
from model import ResNet, ResidualBlock
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import pickle
from PIL import Image

vocab_path = './data/vocab.pkl'
root_path = 'data/bitmap2svg_samples2/'
batch_size = 128
num_workers = 2
crop_size = 64
transform = transforms.Compose([ 
    transforms.ToTensor()
    ])

    
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

data_loader = get_loader(root_path, vocab, 
                     transform, batch_size,
                     shuffle=True, num_workers=num_workers) 


def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([64, 64], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)



bitmap_path = 'data/bitmap2svg_samples2/bitmap/'
bitmap_list = os.listdir(bitmap_path)
cnt = 0

tot_R = [] 
tot_G = []
tot_B = [] 

for element in bitmap_list:
    #cnt+=1 
    #if cnt > 2: 
    #    break;
    file_name = bitmap_path + element
    image = load_image(file_name, transform)
    image_tensor = Variable(image).squeeze()
    R_tensor = image_tensor[0].view(-1).data.numpy()
    G_tensor = image_tensor[1].view(-1).data.numpy()
    B_tensor = image_tensor[2].view(-1).data.numpy()


    tot_R = np.append(tot_R, R_tensor)
    tot_G = np.append(tot_G, G_tensor)
    tot_B = np.append(tot_B, B_tensor)


R_mean = np.mean(tot_R)
R_var = np.var(tot_R)
G_mean = np.mean(tot_G)
G_var = np.var(tot_G)
B_mean = np.mean(tot_B)
B_var = np.var(tot_B)

print(R_mean)
print(G_mean)
print(B_mean)

print(R_var)
print(G_var)
print(B_var)