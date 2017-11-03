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

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda(1)
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
    # For normalization, see https://github.com/pytorch/vision#models
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
    
    # Build data loader
    data_loader = get_loader(args.root_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the models
    #encoder = EncoderCNN(args.embed_size)
    encoder = ResNet(ResidualBlock, [3, 3, 3], len_vocab)
    decoder = DecoderRNN(len_vocab, args.hidden_size, 
                         len(vocab), args.num_layers)

    
    if torch.cuda.is_available():
            encoder.cuda(1)
            decoder.cuda(1)


    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    #params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the Models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):

            #if i > 1 : 
            #   break;

            # make one hot 
            cap_ = torch.unsqueeze(captions,2)
            one_hot_ = torch.FloatTensor(captions.size(0),captions.size(1),len_vocab).zero_()
            one_hot_caption = one_hot_.scatter_(2, cap_, 1)

            # Set mini-batch dataset
            images = to_var(images)  
            captions = to_var(captions)
            captions_ = to_var(one_hot_caption)
            #print(captions_)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]            
            # Forward, Backward and Optimize
            optimizer.zero_grad()
            features = encoder(images)
            outputs = decoder(features, captions_, lengths)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 

                ##test set accuracy 
                #rearrange tensor to batch_size * caption_size 
                re_target = rearrange_tensor(targets, captions.size(0), captions.size(1))
                re_out_max = rearrange_tensor(outputs.max(1)[1], captions.size(0), captions.size(1))
                #convert to numpy 
                outputs_np = re_out_max.cpu().data.numpy()
                targets_np = re_target.cpu().data.numpy()


                location_match = 0 
                exact_match = 0             
                for i in range(len(targets_np)):
                    #print(outputs_np[i])
                    #print('target')
                    #print(targets_np[i])
                    if(outputs_np[i][1] == targets_np[i][1]):
                        location_match +=1
                    if(np.array_equal(outputs_np[i], targets_np[i])):
                        exact_match +=1 
                print('location match accuracy: %.4f, exact match accuracy: %4.f'
                 %(location_match/len(targets_np), exact_match/len(targets_np)))

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
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=64 ,
                        help='size for randomly cropping images')
    parser.add_argument('--root_path', type=str, default='data/bitmap2svg_samples2/',
                        help='path for root')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=5,
                        help='step size for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)