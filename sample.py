import torch
import matplotlib.pyplot as plt
import argparse
import pickle 
from torch.autograd import Variable 
from torchvision import transforms 
from data_loader import build_vocab 
from model import EncoderCNN, DecoderRNN
from model import ResNet, ResidualBlock
from attn_model import ResidualBlock, AttnEncoder, AttnDecoderRnn
from PIL import Image
from SAN_model import SANDecoder


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda(1)
    return Variable(x, volatile=volatile)

def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([64, 64], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image
    
def main(args):
    # Image preprocessing
    transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.033, 0.032, 0.033), 
                             (0.027, 0.027, 0.027))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    len_vocab = vocab.idx 

    # Build Models
    encoder = ResNet(ResidualBlock, [3, 3, 3], len_vocab)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)

    decoder = DecoderRNN(len_vocab, args.hidden_size, 
                         len(vocab), args.num_layers)
    
    attn_encoder = AttnEncoder(ResidualBlock, [3, 3, 3])
    attn_encoder.eval()
    attn_decoder = SANDecoder(args.feature_size, args.hidden_size, 
                         len(vocab), args.num_layers)

    # Load the trained model parameters
    attn_encoder.load_state_dict(torch.load(args.encoder_path))
    attn_decoder.load_state_dict(torch.load(args.decoder_path))


    # Prepare Image
    image = load_image(args.image, transform)
    image_tensor = to_var(image, volatile=True)

    # If use gpu
    if torch.cuda.is_available():
        attn_encoder.cuda(1)
        attn_decoder.cuda(1)
    
    # Generate caption from image
    feature = attn_encoder(image_tensor)
    sampled_ids = attn_decoder.sample(feature)
    ids_arr = []
    for element in sampled_ids: 
        temp = element.cpu().data.numpy()
        ids_arr.append(int(temp))


    
    # Decode word_ids to words
    sampled_caption = []
    for word_id in ids_arr:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out image and generated caption.
    print (sentence)
    #image = Image.open(args.image)
    #plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                        help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='./models/san/3object/encoder-20-150.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/san/3object/decoder-20-150.pkl',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./data/san/vocab3.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--root_path', type=str, default='data/3object/',
                        help='path for root')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=128,
                        help='dimension of word embedding vectors')
    parser.add_argument('--feature_size', type=int , default=128,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=256,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    args = parser.parse_args()
    main(args)