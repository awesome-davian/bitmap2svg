import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.autograd import Variable 
from torchvision import transforms
import pickle
from PIL import Image
from data_loader import build_vocab, get_loader
import numpy as np 
from xml.dom import minidom
import random 
import math 

vocab_path = '../data/vocab.pkl'
root_path = '../data/bitmap2svg_samples2/'
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


def compute_norm_factor():

    bitmap_path = '../data/bitmap2svg_samples2/bitmap/'
    bitmap_list = os.listdir(bitmap_path)

    tot_R = [] 
    tot_G = []
    tot_B = [] 

    for element in bitmap_list:
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


def compute_distance_of_image():

    root_path = '../data/3object_test/'
    save_directory = 'san/' 
    trg_bitmap_dir = root_path + 'bitmap/'
    bitmap_from_trg = root_path + save_directory + 'bitmap_from_trg_svg/'  #bitmap from svg from target caption
    bitmap_from_out = root_path + save_directory + 'bitmap_from_out_cap/'   #bitmap from out caption 


    test_list = os.listdir(trg_bitmap_dir)
    cnt = 0
    tot_dist = 0 
    for fname in test_list: 
        cnt+=1
        #if cnt >2:
        #    break;
        scaled_img_path = bitmap_from_trg + fname
        gen_image_path = bitmap_from_out + fname 
        scaled_image = load_image(scaled_img_path, transform).view(3,-1)
        gen_image = load_image(gen_image_path, transform).view(3,-1)

        dist = F.pairwise_distance(scaled_image, gen_image, p=2).view(-1)
        dist_arr = Variable(dist).data.cpu().numpy()
        dist_value = np.mean(dist_arr)
        tot_dist += dist_value
        #print(dist_value/scaled_image.size(1))

    print(tot_dist/cnt)

    # temp_file = '466.png'
    # scaled_img_path = bitmap_from_trg + temp_file
    # gen_image_path = bitmap_from_out + temp_file 
    # scaled_image = load_image(scaled_img_path, transform).view(3,-1)
    # gen_image = load_image(gen_image_path, transform).view(3,-1)

    # dist = F.pairwise_distance(scaled_image, gen_image, p=2).view(-1)
    # dist_arr = Variable(dist).data.cpu().numpy()
    # dist_value = np.mean(dist_arr)
    
    # print(dist_value)




