import torch
import argparse
import pickle 
from torch.autograd import Variable 
from torchvision import transforms 
from attn_model import ResidualBlock, AttnEncoder, AttnDecoderRnn
from PIL import Image
import os 
from xml.dom import minidom
import cairosvg
from SAN_model import SANDecoder
from utils.gen_bitmap_caption_piechart import PieChartGenerator


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

def parse_caption(cap_arr, doc):

    color_list = ['red', 'orange', 'yellow', 'lime', 'green', 'spring_green', 'cyan', 
                  'skyblue','blue', 'purple', 'pink', 'deep_pink']
    svg_color = ['red', 'orange', 'yellow', 'greenyellow', 'lime', 'springgreen', 'cyan',
                    'blue', 'mediumblue', 'purple', 'pink', 'deeppink']


    if cap_arr[0] == 'circle':
        polygon = doc.createElement('circle')
        polygon.setAttribute('class', 'circle')
        polygon.setAttribute('cx', str(int(cap_arr[1])*50))
        polygon.setAttribute('cy', str(int(cap_arr[2])*50))
        polygon.setAttribute('r', cap_arr[3])

        #change color to hsl
        hsl = color_list.index(cap_arr[4])
        #hsl = str(int(hsl) * 30 )
        #style = "fill: hsl(" + hsl + ",100%,50%);"
        style = "fill: " + svg_color[hsl] + ";"
        polygon.setAttribute('style', style)

        cap_arr = cap_arr[5:]

    elif cap_arr[0] == 'rect':
        polygon = doc.createElement('rect')
        polygon.setAttribute('class', 'rect')
        polygon.setAttribute('x', str(int(cap_arr[1])*50))
        polygon.setAttribute('y', str(int(cap_arr[2])*50))
        polygon.setAttribute('width', cap_arr[3])
        polygon.setAttribute('height', cap_arr[4])

        #change color to hsl
        hsl = color_list.index(cap_arr[5])
        #hsl = str(int(hsl) * 30 )
        #style = "fill: hsl(" + hsl + ",100%,50%);"
        style = "fill: "+ svg_color[hsl]+ ";"
        polygon.setAttribute('style', style)

        cap_arr = cap_arr[6:]
 

    return polygon, cap_arr


def gen_caption_from_image(image_tensor, encoder, decoder, vocab):
        
    # Generate caption from image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
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
    in_caption = sampled_caption[1:-1]
    in_sentence = ' '.join(in_caption)

    return in_sentence


def gen_svg_conv2bitmap(trg_caption):

    #make svg 
    doc = minidom.Document()
    svg = doc.createElement('svg')
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
    svg.setAttribute("width", "500")
    svg.setAttribute("height", "500")
    doc.appendChild(svg)

    #parse caption 
    cap_arr = trg_caption.split(" ")
    while len(cap_arr) != 0:
        polygon, cap_arr = parse_caption(cap_arr, doc)
        svg.appendChild(polygon)

    return doc 


def main(args):

    PIEGEN = PieChartGenerator()
    # Image preprocessing
    transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.033, 0.032, 0.033), 
                             (0.027, 0.027, 0.027))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build Models
    encoder = AttnEncoder(ResidualBlock, [3, 3, 3])
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    decoder = AttnDecoderRnn(args.feature_size, args.hidden_size, 
                         len(vocab), args.num_layers)


    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # If use gpu
    if torch.cuda.is_available():
        encoder.cuda(1)
        decoder.cuda(1)


    trg_bitmap_dir = args.root_path + 'bitmap/'
    trg_cap_dir = args.root_path + 'caption/'
    save_directory = 'gen/'
    out_cap_dir = args.root_path + save_directory + 'caption/'              #output caption from model
    svg_from_trg = args.root_path + save_directory + 'svg_from_trg_caption/'    #svg from target caption 
    svg_from_out = args.root_path + save_directory + 'svg_from_out_caption/'   # svg from output caption 
    bitmap_from_trg = args.root_path + save_directory + 'bitmap_from_trg_svg/'  #bitmap from svg from target caption
    bitmap_from_out = args.root_path + save_directory + 'bitmap_from_out_cap/'   #bitmap from out caption 

    if not os.path.exists(out_cap_dir):
        os.makedirs(out_cap_dir)
    if not os.path.exists(svg_from_trg):
        os.makedirs(svg_from_trg)
    if not os.path.exists(bitmap_from_trg):
        os.makedirs(bitmap_from_trg)
    if not os.path.exists(bitmap_from_out):
        os.makedirs(bitmap_from_out)
    if not os.path.exists(svg_from_out):
        os.makedirs(svg_from_out)

    test_list = os.listdir(trg_bitmap_dir)
    cnt = 0
    for fname in test_list: 
        #if cnt >2:
        #    break;
        #load image 
        try:
            test_path = trg_bitmap_dir + fname
            test_image = load_image(test_path, transform)
            image_tensor = to_var(test_image)

            #gen caption and write to file 
            in_sentence = gen_caption_from_image(image_tensor, encoder, decoder, vocab)
            with open(os.path.join(out_cap_dir, fname), 'w+') as f:
                f.write(in_sentence)        
            cap_name = fname.replace('.png', '.svg')  

            if args.image_type == 'polygon':
                #generate svg from trg_caption, convert to bitmap 
                with open(os.path.join(trg_cap_dir, cap_name), 'r') as f:
                    trg_caption = f.read()
                doc = gen_svg_conv2bitmap(trg_caption)
                #write svg 
                with open(os.path.join(svg_from_trg, cap_name), 'w+') as f:
                    f.write(doc.toxml())

                #convert and save as bitmap 
                svg_path = svg_from_trg + cap_name
                bitmap_path = bitmap_from_trg + fname
                cairosvg.svg2png(url=svg_path, write_to=bitmap_path)
                print(cap_name)

                #generate svg from output caption
                out_doc = gen_svg_conv2bitmap(in_sentence)
                with open(os.path.join(svg_from_out, cap_name), 'w+') as f:
                    f.write(out_doc.toxml())
                #conver and save as bitmap
                svg_out_path = svg_from_out + cap_name
                bitmap_out_path = bitmap_from_out + fname
                cairosvg.svg2png(url=svg_out_path, write_to=bitmap_out_path)

            elif args.image_type == 'pie':
                print(cap_name)
                #gen pie_svg from trg caption
                out_doc = PIEGEN.gen_svg_pie_chart_from_caption(in_sentence)
                with open(os.path.join(svg_from_out, cap_name), 'w+') as f:
                    f.write(out_doc.toxml())
                #conver and save as bitmap
                svg_out_path = svg_from_out + cap_name
                bitmap_out_path = bitmap_from_out + fname
                cairosvg.svg2png(url=svg_out_path, write_to=bitmap_out_path)

            cnt +=1  
            print(cnt)
      
        except:
            continue

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_type', type=str, help='image type')
    parser.add_argument('--encoder_path', type=str, default='./models/pie/encoder-20-150.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/pie/decoder-20-150.pkl',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./data/pie.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--root_path', type=str, default='data/piechart_test/',
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