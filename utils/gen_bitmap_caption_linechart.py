import numpy as np
import os
from xml.dom import minidom
import random 
import math 
import cairosvg


class LineChartGenerator():

    def __init__(self):
        super(LineChartGenerator, self).__init__()



    def gen_line_chart(self, in_caption=None):
        #make svg 
        doc = minidom.Document()
        svg_width = "1000"
        svg_height = "500"
        svg = doc.createElement('svg')
        svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
        svg.setAttribute("width", svg_width)
        svg.setAttribute("height", svg_height)
        doc.appendChild(svg)

        g = doc.createElement('g')
        g.setAttribute("transform", "translate(50,20)")
        svg.appendChild(g)
        data = {}
        #set num of element, portion 
       
        if in_caption == None:
            num_element = random.randrange(6,30)
            scale_factor = random.randrange(1,3)
            svg_color = ['red', 'blue', 'purple', 'black', "green", "orange"]
            rand_color_idx = random.randrange(0, len(svg_color))
            data_arr =  np.random.random_integers(1, math.pow(10,scale_factor), num_element)
            data['scale_factor'] = math.pow(10,scale_factor - 1 )
            data['data_arr'] = data_arr
            data['color'] = svg_color[rand_color_idx]
            data['chart_height'] = "450"
            data['chart_width'] = "890"

        else:
            data['color'] = in_caption[1]
            data['scale_factor'] = in_caption[2]
            data['max'] = in_caption[3]
            data_arr = [] 
            for element in in_caption[5:]:
                value = int(element) * int(data['max']) / 450
                data_arr.append(value)
            data['data_arr'] = data_arr
            data['chart_height'] = "450"
            data['chart_width'] = "890"

        # x axis 
        x_axis = self.create_axis(doc, data, axis_type='x')
        g.appendChild(x_axis)
        # y axis 
        y_axis = self.create_axis(doc, data, axis_type='y')
        g.appendChild(y_axis)
        # calculate element 
        line, caption = self.calculate_line(doc, data)
        g.appendChild(line)


        return doc, caption 



    def calculate_line(self, doc, data):

        data_arr = data['data_arr']
        scale_factor = int(data['scale_factor'])
        color = data['color']
        chart_height = int(data['chart_height'])
        chart_width = int(data['chart_width'])


        lines = doc.createElement("path")
        lines.setAttribute("class", "line")


        lines.setAttribute("stroke", color)
        lines.setAttribute("stoke-width", "1.5")
        lines.setAttribute("fill", "none")
        lines.setAttribute("stoke-linejoin", "round")
        lines.setAttribute("stroke-linecap", "round")

        line_path = "" 
        point_width = chart_width/(len(data_arr)-1)
        max_y = (np.amax(data_arr) // scale_factor + 1)*scale_factor 

        caption = [] 
        caption.append('linechart')
        caption.append(color)
        caption.append(str(scale_factor))
        caption.append(str(max_y))
        caption.append('data')

        for idx, element in enumerate(data_arr):
        	if idx == 0 : 
        		line_path += "M0"+","+str(chart_height - chart_height*element/max_y)
        	else: 
        		line_path += "L"+str(point_width*idx)+","+str(chart_height - chart_height*element/max_y)

        lines.setAttribute("d", line_path)

        for element	in data_arr:
        	element = str(int(element))
        	caption.append(element)

        return lines, caption



    def create_axis(self, doc, data, axis_type):

        data_arr = data['data_arr']
        scale_factor = int(data['scale_factor'])
        chart_height = int(data['chart_height'])
        chart_width = int(data['chart_width'])

        axis_g = doc.createElement('g')
        axis_g.setAttribute("fill", "none")
        axis_g.setAttribute("font-size", "10")

        domain = doc.createElement("path")
        domain.setAttribute("class", "domain")
        domain.setAttribute("stroke", "#000")

    	
        if axis_type == 'x':
        	x_label_arr = ['A', 'B', 'C', 'D', 'E', 'F']
        	tick_wide = chart_width/(len(x_label_arr)+1)
        	
        	axis_g.setAttribute("transform", "translate(0,450)")
        	axis_g.setAttribute("text-anchor", "middle")

        	domain.setAttribute("d", "M0.5,6V0H890.5V6")
        	axis_g.appendChild(domain)

        	for idx, element in enumerate(x_label_arr):
        		axis_tick = doc.createElement('g')
        		axis_tick.setAttribute("class", "tick")
        		axis_tick.setAttribute("opacity", "1")
        		axis_tick.setAttribute("transform", "translate("+str(tick_wide*(idx+1))+",0)")
        		
        		line = doc.createElement("line")
        		line.setAttribute("stroke", "#000")
        		line.setAttribute("y2", "6")
        		axis_tick.appendChild(line)

        		text_holder = doc.createElement("text")
        		text_holder.setAttribute("fill","#000")
        		text_holder.setAttribute("y", "18")
        		text = doc.createTextNode(x_label_arr[idx])
        		text_holder.appendChild(text)
        		axis_tick.appendChild(text_holder)

        		axis_g.appendChild(axis_tick)

        elif axis_type == 'y':
        	max_y = int(np.amax(data_arr) // scale_factor)
        	tick_wide = chart_height/(max_y + 1)
        	axis_g.setAttribute("text-anchor", "end")

        	domain.setAttribute("d", "M-6,450.5H0.5V0.5H-6")
        	axis_g.appendChild(domain)

        	for i in range(1, max_y + 1):
        		axis_tick = doc.createElement("g")
        		axis_tick.setAttribute("class", "tick")
        		axis_tick.setAttribute("opacity", "1")
        		axis_tick.setAttribute("transform", 
        			"translate(0,"+ str(chart_height - tick_wide*i) +")")
        		
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

    LINEGEN = LineChartGenerator()

    root_path = '../data/linechart_test/'
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


    # doc, caption = LINEGEN.gen_line_chart()
    # with open(os.path.join(trg_svg_dir, '1.svg'), 'w+') as f:
    # 	f.write(doc.toxml())


    for i in range(1000):
        if i%100 == 0:
            print(i) 

        doc, caption = LINEGEN.gen_line_chart()
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