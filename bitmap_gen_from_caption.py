import os 
from xml.dom import minidom
import argparse
import cairosvg

def parse_caption(cap_arr, doc):

	color_list = ['red', 'orange', 'yellow', 'lime', 'green', 'spring_green', 'cyan', 
	              'skyblue','blue', 'purple', 'pink', 'deep_pink']

	if cap_arr[0] == 'circle':
		polygon = doc.createElement('circle')
		polygon.setAttribute('class', 'circle')
		polygon.setAttribute('cx', cap_arr[1])
		polygon.setAttribute('cy', cap_arr[2])
		polygon.setAttribute('r', cap_arr[3])

		#change color to hsl
		hsl = color_list.index(cap_arr[4])
		hsl = str(int(hsl) * 30 )
		style = "fill: hsl(" + hsl + ",100%,50%);"
		polygon.setAttribute('style', style)

		cap_arr = cap_arr[5:]

	elif cap_arr[0] == 'rect':
		polygon = doc.createElement('rect')
		polygon.setAttribute('class', 'rect')
		polygon.setAttribute('x', cap_arr[1])
		polygon.setAttribute('y', cap_arr[2])
		polygon.setAttribute('width', cap_arr[3])
		polygon.setAttribute('height', cap_arr[4])

		#change color to hsl
		hsl = color_list.index(cap_arr[5])
		hsl = str(int(hsl) * 30 )
		style = "fill: hsl(" + hsl + ",100%,50%);"
		polygon.setAttribute('style', style)

		cap_arr = cap_arr[6:]

	return polygon, cap_arr



def main(args):

	root_path = args.root_path
	cap_path = root_path + 'caption/'
	svg_from_trg = root_path + 'gen/svg_from_caption/'
	bitmap_from_trg = root_path + 'gen/bitmap_from_caption/'
	if not os.path.exists(svg_from_trg):
		os.makedirs(svg_from_trg)
	if not os.path.exists(bitmap_from_trg):
		os.makedirs(bitmap_from_trg)

	file_list = os.listdir(cap_path)
	cnt = 0 

	for f_list in file_list:
		if cnt >2:
			break;
		#cnt+=1 
		#read caption 
		with open(os.path.join(cap_path, f_list), 'r') as f:
			caption = f.read()

		#make svg 
		doc = minidom.Document()
		svg = doc.createElement('svg')
		svg.setAttribute("xmlns", "http://www.w3.org/2000")
		svg.setAttribute("width", "500")
		svg.setAttribute("height", "500")
		doc.appendChild(svg)

		#parse caption 
		cap_arr = caption.split(" ")
		while len(cap_arr) != 0:
			polygon, cap_arr = parse_caption(cap_arr, doc)
			svg.appendChild(polygon)

        #write svg 
		with open(os.path.join(svg_from_trg, f_list), 'w+') as f:
			f.write(doc.toxml())

        #convert and save as bitmap 
		svg_path = svg_from_trg + f_list
		bitmap_path = bitmap_from_trg + f_list.replace('.svg', '.png')
		cairosvg.svg2png(url=svg_path, write_to=bitmap_path)










if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/nobject/caption', 
                        help='path for train annotation file')

    parser.add_argument('--svg_path', type=str, 
                        default='data/nobject/svg', 
                        help='ath for train annotation file')

    parser.add_argument('--root_path', type=str, 
                        default='data/nobject_test/', 
                        help='root path for train annotation file')

    args = parser.parse_args()
    main(args)