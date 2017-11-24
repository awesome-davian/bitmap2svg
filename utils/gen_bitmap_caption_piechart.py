import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
from PIL import Image
import numpy as np 
from xml.dom import minidom
import random 
import math 
import cairosvg

class PieChartGenerator():

    def __init__(self):
        super(PieChartGenerator, self).__init__()


    def caption_parser(self, caption):

        caption_arr = caption.split(' ')
        radius = caption_arr[1]
        caption_arr = caption_arr[2:]
        portion_arr = [] 
        color_arr = [] 
        for idx, element in enumerate(caption_arr):
            if idx % 3 == 1 :
                portion_arr.append(float(element))
            elif idx % 3 ==2 :
                color_arr.append(element)

        return int(radius), portion_arr, color_arr



    def gen_svg_pie_chart_from_caption(self, caption):
        #make svg 
        doc = minidom.Document()
        svg = doc.createElement('svg')
        svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
        svg.setAttribute("width", "500")
        svg.setAttribute("height", "500")
        doc.appendChild(svg)

        g = doc.createElement('g')
        g.setAttribute("transform", "translate(10,10)")
        svg.appendChild(g)

        radius, portion_arr, color_arr = self.caption_parser(caption)

        sectors = self.calculateSectors(portion_arr, color_arr, radius)
        for sector in sectors:
            arc = self.make_arc(doc, sector)
            g.appendChild(arc)

        return doc 

    def gen_svg_pie_chart(self):
        #make svg 
        doc = minidom.Document()
        svg = doc.createElement('svg')
        svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
        svg.setAttribute("width", "500")
        svg.setAttribute("height", "500")
        doc.appendChild(svg)

        g = doc.createElement('g')
        g.setAttribute("transform", "translate(10,10)")
        svg.appendChild(g)
        
        #set num of element, portion 
        num_element = random.randrange(4,10)
        portion =  np.random.random_integers(1, 10, num_element)
        portion = np.around(portion* 1/np.sum(portion), decimals =2)
        portion[-1] += 1- np.sum(portion)
        
        # calculate sectors
        sectors = self.calculateSectors(portion)

        caption = [] 
        for idx, sector in enumerate(sectors):
            arc = self.make_arc(doc,sector)
            g.appendChild(arc)

            if idx == 0:
                caption.append('piechart')
                caption.append(sector['radius'])
                caption = self.get_caption(sector, caption)
            else:
                caption = self.get_caption(sector, caption)

        return doc, caption 

    def get_caption(self, sector, caption):

        caption.append('arc')
        caption.append(sector['percent'])
        caption.append(sector['color'])


        return caption


    def make_arc(self,doc, sector):
        arc = doc.createElement('g')
        arc.setAttribute("class", "arc")
        path = doc.createElement('path')
        path.setAttribute('d', 'M' + sector['l'] + ',' + sector['l'] + ' L' +
        	sector['l'] + ',0 A' + sector['l']+ ',' + sector['l'] + ' 1 0,1 ' + sector['x'] + ', ' + sector['y'] + ' z')
        path.setAttribute('transform', 'rotate(' + sector['rotation'] + ', '+ sector['l']+', '+ sector['l']+')')
        path.setAttribute('style', 'fill: ' +sector['color'] + ';')
        arc.appendChild(path)

        return arc 

    def calculateSectors(self, portion, color_arr=None, radius=None):

        sectors = []
        if radius is None:
            radius = random.randrange(100,200)
        if color_arr is None:
            svg_color = ['red', 'orange', 'yellow', 'greenyellow', 'lime', 'springgreen', 'cyan',
                    'blue', 'mediumblue', 'purple', 'pink', 'deeppink']
            color_arr = np.random.choice(svg_color, len(portion), replace=False)

        label_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        label_arr = np.random.choice(label_char, len(portion),  replace=False)

        rotation = 0 
        for idx, element in enumerate(portion):
            a = 360 * element
            x = 0.0
            arc_sweep = 0 
            x_ = 0.0

            if a > 180:
                a = 360 - a

            aRad = math.pi * a / 180 
            z = math.sqrt(2 * math.pow(radius,2) - (2 * math.pow(radius,2) * math.cos(aRad)))
            
            if a <= 90 :
                x = radius * math.sin(aRad)
            else: 
                x = radius*math.sin((180 - a) * math.pi/180 )
            
            
            y = math.sqrt( z*z - x*x )
            
            if a <= 180: 
            	x_ = radius + x
            	arc_sweep = 0
            else:
                x_ = radius - x
                arc_sweep = 1

            sector = {}
            sector['radius'] = str(radius)
            sector['percent'] = str(element)
            sector['label'] = label_arr[idx] 
            sector['color'] = color_arr[idx]
            sector['arc_sweep'] = str(arc_sweep)
            sector['l'] = str(radius)
            sector['x'] = str(x_)
            sector['y'] = str(y)
            sector['rotation'] = str(rotation)
            sectors.append(sector)

            rotation += a

        return sectors


def main():

    PIEGEN = PieChartGenerator()

    root_path = '../data/piechart_test/'
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

    for i in range(1000):

        try:
            if i%100 == 0:
                print(i) 


            doc, caption = PIEGEN.gen_svg_pie_chart()
            svg_name = str(i) + '.svg'
            bitmap_name = str(i) + '.png'
            with open(os.path.join(trg_svg_dir, svg_name), 'w+') as f:
                f.write(doc.toxml())

            with open(os.path.join(trg_caption_dir, svg_name), 'w+') as f:
                cap_str = ' '.join(caption)
                f.write(cap_str)

            cairosvg.svg2png(url=trg_svg_dir + svg_name, write_to= trg_bitmap_dir + bitmap_name)
        
        except:
            continue

if __name__ == '__main__':

    main()