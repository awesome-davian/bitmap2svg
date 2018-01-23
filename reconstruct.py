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
import numpy as np
import cv2

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


def parse_predict_sentence(predict):

	predict_caption = predict
	data_arr = [] 
	class_arr = [] 
	rgb_arr = [] 
	while len(predict_caption) != 0:
		if predict_caption[0] == 'circle':
			class_arr.append('circle')
			data_arr.append(predict_caption[1:4])
			rgb_arr.append(predict_caption[4:7])
			predict_caption = predict_caption[7:]
		elif predict_caption[0] == 'rect':
			class_arr.append('rect')
			data_arr.append(predict_caption[1:5])
			rgb_arr.append(predict_caption[5:8])
			predict_caption = predict_caption[8:]			
		elif predict_caption[0] == 'line':
			class_arr.append('line')
			data_arr.append(predict_caption[1:6])
			rgb_arr.append(predict_caption[6:9])
			predict_caption = predict_caption[9:]

	return data_arr, class_arr, rgb_arr		

def format_color(r,g,b):
    return '#{:02x}{:02x}{:02x}'.format(r,g,b)


def gen_svg_from_predict(predict,image):

	doc = minidom.Document()
	svg_width = '500'
	svg_height = '500'
	svg = doc.createElement('svg')
	svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
	svg.setAttribute("width", svg_width)
	svg.setAttribute("height", svg_height)
	doc.appendChild(svg)

	data_arr, class_arr, rgb_arr = parse_predict_sentence(predict)
	for i, element in enumerate(data_arr):

		try:
			if class_arr[i] == 'circle':
				r = (int(element[0]) -1 ) * 4 
				cx = (int(element[1]) -1 ) * 4 
				cy = (int(element[2]) -1 ) * 4 
				polygon = doc.createElement('circle')
				polygon.setAttribute('r',str(int(r)))
				polygon.setAttribute('cx', str(cx))
				polygon.setAttribute('cy', str(cy))
				color = format_color(int(rgb_arr[i][0]),
					int(rgb_arr[i][1]), int(rgb_arr[i][2]))
				polygon.setAttribute('style', 'fill:'+color+';')

			elif class_arr[i]== 'rect':
				x = (int(element[0]) -1 ) * 4 
				y = (int(element[1]) -1 ) * 4 
				width = (int(element[2]) -1 ) * 4 
				height = (int(element[3]) -1 ) * 4 
				polygon = doc.createElement('rect')
				polygon.setAttribute('width', str(width))
				polygon.setAttribute('height', str(height))
				polygon.setAttribute('x', str(x))
				polygon.setAttribute('y', str(y))
				color = format_color(int(rgb_arr[i][0]),
					int(rgb_arr[i][1]), int(rgb_arr[i][2]))
				polygon.setAttribute('style','fill:'+color+';')

			elif class_arr[i] == 'line':
				# x_1 = (int(element[0]) -1 ) * 4 
				# y_1 = (int(element[1]) -1 ) * 4 
				# x_2 = (int(element[2]) -1 ) * 4 
				# y_2 = (int(element[3]) -1 ) * 4 
				# x_min = min(x_1, x_2)
				# x_max = max(x_1, x_2)
				# y_min = min(y_1, y_2)
				# y_max = max(y_1, y_2)

				x_min = (int(element[0]) -1 ) * 4 
				y_min = (int(element[1]) -1 ) * 4 
				x_max = (int(element[2]) -1 ) * 4 
				y_max = (int(element[3]) -1 ) * 4 
				# crop_img = image[int(y_min):int(y_max), int(x_min):int(x_max)]
				# crop_img = np.sum(crop_img, axis=2)
				# vertical_indicies = np.where(np.any(crop_img, axis=1))[0]
				# mask_y_min = vertical_indicies[0]
				# mask_y_max = vertical_indicies[-1]
				# x_y_min = np.where(crop_img[mask_y_min,:]>0)[0][0]
				# x_y_max = np.where(crop_img[mask_y_max,:]>0)[0][0]
				# polygon = doc.createElement('line')
				# if x_y_max > x_y_min:
				# 	polygon.setAttribute('x1', str(x_min))
				# 	polygon.setAttribute('y1', str(y_min))
				# 	polygon.setAttribute('x2', str(x_max))
				# 	polygon.setAttribute('y2', str(y_max))		
				# else:
				# 	polygon.setAttribute('x1', str(x_min))
				# 	polygon.setAttribute('y1', str(y_max))
				# 	polygon.setAttribute('x2', str(x_max))
				# 	polygon.setAttribute('y2', str(y_min))	 
				polygon = doc.createElement('line')
				polygon.setAttribute('x1', str(x_min))
				polygon.setAttribute('y1', str(y_min))
				polygon.setAttribute('x2', str(x_max))
				polygon.setAttribute('y2', str(y_max))			
				color = format_color(int(rgb_arr[i][0]),
					int(rgb_arr[i][1]), int(rgb_arr[i][2]))
				polygon.setAttribute('style', 'stroke:'+color+';stroke-width:'+str(4))

			svg.appendChild(polygon)

		except:
			continue		

	return doc.toxml()

def main(args):

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
    save_directory = 'predict/'
    svg_from_out = args.root_path + save_directory + 'svg/'   # svg from output caption 
    bitmap_from_out = args.root_path + save_directory + 'bitmap/'   #bitmap from out caption 

    if not os.path.exists(bitmap_from_out):
        os.makedirs(bitmap_from_out)
    if not os.path.exists(svg_from_out):
        os.makedirs(svg_from_out)

    test_list = os.listdir(trg_bitmap_dir)
    for i, fname in enumerate(test_list): 
        print(fname)
        test_path = trg_bitmap_dir + fname
        test_image = load_image(test_path, transform)
        image_tensor = to_var(test_image)
        in_sentence = gen_caption_from_image(image_tensor, encoder, decoder, vocab)
        print(in_sentence)
        image_matrix = cv2.imread(test_path)
        doc = gen_svg_from_predict(in_sentence.split(' '), image_matrix)

        with open(os.path.join(svg_from_out, fname.split('.')[0]+'.svg'), 'w+') as f:
            f.write(doc)
        cairosvg.svg2png(url=svg_from_out+ fname.split('.')[0] + '.svg', write_to= bitmap_from_out+fname)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='./models/polygon_n/encoder-60-780.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/polygon_n/decoder-60-780.pkl',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./data/polygon_n.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--root_path', type=str, default='dataset/polygon_test/',
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