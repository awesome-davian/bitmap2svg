import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from data_loader import build_vocab, get_loader
from model import ResNet, ResidualBlock
from attn_model import ResidualBlock, AttnEncoder, AttnDecoderRnn
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import pickle

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def rearrange_tensor(x, batch_size, caption_size):
    for i in range(caption_size):
        temp = x[i*batch_size:(i+1)*batch_size].view(batch_size, -1)
        if i == 0:
            temp_cat = temp 
        else: 
            temp_cat = torch.cat((temp_cat,  temp), 1)

    return temp_cat


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.033, 0.032, 0.033), 
                             (0.027, 0.027, 0.027))])
    
    # Build vocab  
    vocab = build_vocab(args.root_path, threshold=0)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    len_vocab = vocab.idx
    print(vocab.idx2word)
    
    # Build data loader
    data_loader = get_loader(args.root_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the attn models
    encoder = AttnEncoder(ResidualBlock, [3, 3, 3])
    decoder = AttnDecoderRnn(args.feature_size, args.hidden_size, 
                         len(vocab), args.num_layers)

    
    if torch.cuda.is_available():
            encoder.cuda()
            decoder.cuda()


    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the Models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):

            #if i > 1 : 
             # break;

            # to variable 
            images = to_var(images)  
            captions = to_var(captions)
            
            # Forward, Backward and Optimize
            optimizer.zero_grad()
            features = encoder(images)
            outputs = decoder(features, captions, lengths)

            captions = captions.view(-1)
            outputs = outputs.view(-1,len_vocab)

            loss = criterion(outputs, captions)
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 

                #test set accuracy 
                outputs_np = outputs.max(1)[1].cpu().data.numpy()
                targets_np = captions.cpu().data.numpy()

                #print(outputs_np)
                #print(targets_np)

                location_match = 0 
                size_match = 0   
                shape_match = 0 
                exact_match = 0           
                for i in range(len(targets_np)):
                    if outputs_np[i] == targets_np[i]:
                        exact_match +=1 
                    if i >= args.batch_size and i < args.batch_size*2 and outputs_np[i] == targets_np[i]:
                        shape_match +=1 
                    elif i >= args.batch_size*2 and i < args.batch_size*3 and outputs_np[i] == targets_np[i]:
                        location_match +=1
                    elif i >= args.batch_size*3 and i < args.batch_size*4 and outputs_np[i] == targets_np[i]:
                        size_match  +=1

                print('location match : %.4f, shape match : %.4f, exact_match: %.4f'
                 %(location_match/(args.batch_size), shape_match/args.batch_size, exact_match/len(targets_np)))

            # Save the models
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/attn/1object/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=128,
                        help='size for randomly cropping images')
    parser.add_argument('--root_path', type=str, default='data/circle_and_rect/',
                        help='path for root')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=50,
                        help='step size for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='./data/attn/vocab1.pkl', 
                        help='path for saving vocabulary wrapper')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--feature_size', type=int , default=256 ,
                        help='dimension of feature embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)