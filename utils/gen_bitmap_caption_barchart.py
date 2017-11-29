import numpy as np
import os
from xml.dom import minidom
import random 
import math 
import cairosvg


class BarChartGenerator():

    def __init__(self):
        super(BarChartGenerator, self).__init__()

    def gen_bar_chart(self, in_caption=None):
        #make svg 
        doc = minidom.Document()
        svg_width = '800'
        svg_height = '500'
        svg = doc.createElement('svg')
        svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
        svg.setAttribute("width", str(svg_width))
        svg.setAttribute("height", str(svg_height))
        doc.appendChild(svg)

        g = doc.createElement('g')
        g.setAttribute("transform", "translate(50,20)")
        svg.appendChild(g)
        data = {}
        #set num of element, portion 
       
        if in_caption == None:
        	num_element = random.randrange(4,10)
        	scale_factor = random.randrange(1,3)
        	svg_color = ['red', 'blue', 'purple', 'black', "green", "orange"]
        	rand_color_idx = random.randrange(0, len(svg_color))
        	data_arr =  np.random.random_integers(1, math.pow(10,scale_factor), num_element)
        	label_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']	
        	label_arr = np.random.choice(label_char, num_element,  replace=False)
        	
        	data['scale_factor'] = math.pow(10,scale_factor - 1 )
        	data['data_arr'] = data_arr
        	data['color'] = svg_color[rand_color_idx]
        	data['svg_width'] = svg_width
        	data['svg_height'] =svg_height
        	data['label_arr'] = label_arr.tolist()
        	data['tick_padding'] = '10'

        else:
        	data['color'] = in_caption[1]
        	data['tick_padding'] = in_caption[2]
        	data['scale_factor'] = in_caption[3]
        	data['max'] = in_caption[4]
        	data['svg_width'] = svg_width
        	data['svg_height'] = svg_height
        	data_arr = []
        	label_arr = []

        	for idx, element in enumerate(in_caption[6:]):
        		if idx % 2 == 0:
        			value = int(element) * int(data['max']) / int(svg_height)
        			data_arr.append(value)
        		elif idx % 2 == 1:
        			label_arr.append(element)

        	data['label_arr']  = label_arr
        	data['data_arr'] = data_arr


        # x axis 
        x_axis = self.create_axis(doc, data, axis_type='x')
        g.appendChild(x_axis)
        # y axis 
        y_axis = self.create_axis(doc, data, axis_type='y')
        g.appendChild(y_axis)
        # calculate element 
        bars, caption = self.calculate_bars(doc, data)
        g.appendChild(bars)


        return doc, caption 

    def parse_caption(doc,data_arr):
    	rect = doc.createElement("rect")
    	rect.setAttribute("x", data_arr[1])
    	rect.setAttribute("y", data_arr[2])
    	rect.setAttribute("la")

    def calculate_bars(self, doc, data):

    	data_arr = data['data_arr']
    	scale_factor = int(data['scale_factor'])
    	color = data['color']
    	svg_width = int(data['svg_width']) - 50 
    	tick_padding = int(data['tick_padding'])
    	svg_height = int(data['svg_height']) - 50
    	tick_wide = (svg_width - tick_padding*len(data_arr))/(len(data_arr))
    	data_bar = doc.createElement("g")
    	label_arr = data['label_arr']
    	max_y = (np.amax(data_arr) // scale_factor + 1)*scale_factor 


    	caption = [] 
    	caption.append('barchart')
    	caption.append(color)
    	caption.append(str(tick_padding))
    	caption.append(str(scale_factor))
    	caption.append(str(max_y))
    	caption.append('data')


    	for idx, element in enumerate(data_arr):
    		rect = doc.createElement("rect")
    		rect.setAttribute('style', 'fill: ' +color + ';')
    		rect.setAttribute("width", str(tick_wide))
    		rect.setAttribute("height", str(svg_height*element/max_y))
    		#rect.setAttribute("transform", "translate(0," + str(svg_height) +")")

    		rect.setAttribute("x", str(tick_padding + (tick_wide + tick_padding)*idx))
    		rect.setAttribute("y", str(svg_height - svg_height*element/max_y))

    		caption.append(str(int(element)))
    		caption.append(label_arr[idx])
    		data_bar.appendChild(rect)


    	return data_bar, caption




    def create_axis(self, doc, data, axis_type):

    	data_arr = data['data_arr']
    	scale_factor = int(data['scale_factor'])
    	svg_width = int(data['svg_width']) - 50 
    	svg_height = int(data['svg_height']) - 50 
    	tick_padding = int(data['tick_padding'])
    	label_arr = data['label_arr']

    	axis_g = doc.createElement('g')
    	axis_g.setAttribute("fill", "none")
    	axis_g.setAttribute("font-size", "10")

    	domain = doc.createElement("path")
    	domain.setAttribute("class", "domain")
    	domain.setAttribute("stroke", "#000")


    	
    	if axis_type == 'x':
    		tick_wide = (svg_width - tick_padding*len(data_arr))/(len(data_arr))
    		
    		axis_g.setAttribute("transform", "translate(0," + str(svg_height) +")")
    		axis_g.setAttribute("text-anchor", "middle")

    		domain.setAttribute("d", "M0.5,6V0H"+str(svg_width)+"V6")
    		axis_g.appendChild(domain)

    		for idx, element in enumerate(label_arr):
    			axis_tick = doc.createElement('g')
    			axis_tick.setAttribute("class", "tick")
    			axis_tick.setAttribute("opacity", "1")
    			axis_tick.setAttribute("transform", 
    				"translate("+str((tick_wide+tick_padding)*idx + tick_wide/2 + tick_padding)+",0)")
    		
    			line = doc.createElement("line")
    			line.setAttribute("stroke", "#000")
    			line.setAttribute("y2", "6")
    			axis_tick.appendChild(line)

    			text_holder = doc.createElement("text")
    			text_holder.setAttribute("fill","#000")
    			text_holder.setAttribute("y", "18")
    			text = doc.createTextNode(element)
    			text_holder.appendChild(text)
    			axis_tick.appendChild(text_holder)

    			axis_g.appendChild(axis_tick)

    	elif axis_type == 'y':
    		max_y = int(np.amax(data_arr) // scale_factor)
    		tick_wide = svg_height/(max_y + 1)
    		axis_g.setAttribute("text-anchor", "end")

    		domain.setAttribute("d", "M-6,"+str(svg_height)+"H0.5V0.5H-6")
    		axis_g.appendChild(domain)

    		for i in range(1, max_y + 1):
    			axis_tick = doc.createElement("g")
    			axis_tick.setAttribute("class", "tick")
    			axis_tick.setAttribute("opacity", "1")
    			axis_tick.setAttribute("transform", 
    				"translate(0,"+ str(svg_height - int(tick_wide)*i) +")")
    			
    			line = doc.createElement("line")
    			line.setAttribute("stroke", "#000")
    			line.setAttribute("x2", "-6")
    			axis_tick.appendChild(line)

    			text_holder = doc.createElement("text")
    			text_holder.setAttribute("fill","#000")
    			text_holder.setAttribute("x", "-9")
    			text = doc.createTextNode(str(i*scale_factor))
    			text_holder.appendChild(text)
    			axis_tick.appendChild(text_holder)

    			axis_g.appendChild(axis_tick)


    	return axis_g



def main():

    BARGEN = BarChartGenerator()

    root_path = '../data/barchart/'
    trg_bitmap_dir = root_path + 'bitmap/'
    trg_svg_dir = root_path + 'svg/'
    trg_caption_dir = root_path + 'caption'


    if not os.path.exists((root_path)):
        os.makedirs((root_path))
    if not os.path.exists(trg_bitmap_dir):
        os.makedirs(trg_bitmap_dir)
    if not os.path.exists(trg_svg_dir):
        os.makedirs(trg_svg_dir)
    if not os.path.exists(trg_caption_dir):
        os.makedirs(trg_caption_dir)


    for i in range(100000):
        if i%1000 == 0:
            print(i) 

        doc, caption = BARGEN.gen_bar_chart()
        svg_name = str(i) + '.svg'
        bitmap_name = str(i) + '.png'
        with open(os.path.join(trg_svg_dir, svg_name), 'w+') as f:
            f.write(doc.toxml())

        with open(os.path.join(trg_caption_dir, svg_name), 'w+') as f:
            cap_str = ' '.join(caption)
            f.write(cap_str)


        cairosvg.svg2png(url=trg_svg_dir + svg_name, write_to= trg_bitmap_dir + bitmap_name)

if __name__ == '__main__':

    main()