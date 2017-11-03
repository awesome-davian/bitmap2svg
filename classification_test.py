import argparse
import torch
import torch.nn as nn
import numpy as np
from data_loader import build_vocab, get_loader
from model import ResNet, ResidualBlock
from torch.autograd import Variable 
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

    
    # Image preprocessing
    transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.033, 0.032, 0.033), 
                             (0.027, 0.027, 0.027))])

    vocab = build_vocab(args.root_path, threshold=0)
    num_class = 9

    
    # Build data loader
    data_loader = get_loader(args.root_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the models
    cnn = ResNet(ResidualBlock, [3, 3, 3], num_class)

    
    if torch.cuda.is_available():
            cnn.cuda(1)


    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params =  list(cnn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the Models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):

            #if i > 1 : 
            #  break;
            idx_arr = []
            for element in captions[:,1]:
            	idx_arr.append(int(vocab.idx2word[element]) - 1)
            temp_arr= np.array(idx_arr)
            trg_arr = torch.from_numpy(temp_arr)
            target  = to_var(trg_arr)
            images = to_var(images) 

            optimizer.zero_grad()         
            features = cnn(images)
            loss = criterion(features, target)
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 

                #print(features)
                #print(target)

                ##test set accuracy 
                #rearrange tensor to batch_size * caption_size 
                re_target = rearrange_tensor(target, captions.size(0), 1)
                re_out_max = rearrange_tensor(features.max(1)[1], captions.size(0), 1)
                #convert to numpy 
                outputs_np = re_out_max.cpu().data.numpy()
                targets_np = re_target.cpu().data.numpy()

                location_match = 0 
                for i in range(len(targets_np)):

                    if(outputs_np[i] == targets_np[i]):
                        location_match +=1
                print('location match accuracy: %.4f'
                 %(location_match/len(targets_np)))


    #test model 
    print('---------------------------------')
    cnn.eval()
    test_loader = get_loader(args.test_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 
    for images, captions, lengths in test_loader:
    	idx_arr = []
    	for element in captions[:,1]:
    		idx_arr.append(int(vocab.idx2word[element]) - 1)
    	temp_arr= np.array(idx_arr)
    	trg_arr = torch.from_numpy(temp_arr)
    	target  = to_var(trg_arr)

    	images = to_var(images) 
    	features = cnn(images) 
    
    	re_target = rearrange_tensor(target, captions.size(0), 1)
    	re_out_max = rearrange_tensor(features.max(1)[1], captions.size(0), 1)
    	#convert to numpy 
    	outputs_np = re_out_max.cpu().data.numpy()
    	targets_np = re_target.cpu().data.numpy()

    	location_match = 0 
    	for i in range(len(targets_np)):
    		if(outputs_np[i] == targets_np[i]):
    			location_match +=1
    	print('location match accuracy: %.4f'
    		%(location_match/len(targets_np)))






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--root_path', type=str, default='data/bitmap2svg_samples2/',
                        help='path for root')
    parser.add_argument('--test_path', type=str, default='data/bitmap2_test/',
                        help='path for root')
    parser.add_argument('--log_step', type=int , default=5,
                        help='step size for prining log info')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--hidden_size', type=int , default=128 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)