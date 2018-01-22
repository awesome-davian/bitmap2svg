import numpy as np
import os
from xml.dom import minidom
import random 
import math 
import cairosvg
from PIL import Image 
import cv2
import pickle
from tempfile import TemporaryFile

class SVGGenerator():

    def __init__(self):
        super(SVGGenerator, self).__init__()


    def gen_svg_from_model(self, root_path, element_arr, out_data_arr):
        #make svg 
        doc = minidom.Document()
        svg_width = '500'
        svg_height = '500'
        svg = doc.createElement('svg')
        svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
        svg.setAttribute("width", svg_width)
        svg.setAttribute("height", svg_height)
        doc.appendChild(svg)
        
        for i, element in enumerate(element_arr):
            polygon = self.create_element(doc, element, out_data_arr)
            out_data_arr = out_data_arr[21:]
            svg.appendChild(polygon)
        return doc

    def load_image(self, image_path, transform):
        image = Image.open(image_path).convert('RGB')
        image = image.resize([64, 64], Image.LANCZOS)
        
        if transform is not None:
            image = transform(image).unsqueeze(0)
        
        return image


    def write_and_load_file(self, doc, path, file_name, file_idx, transform):
        save_dir = path + 'gen/'
        step_bitmap_dir = save_dir +'step/bitmap/'+file_name+'/'
        step_svg_dir = save_dir + 'step/svg/'+file_name+'/'

        if not os.path.exists(step_bitmap_dir):
            os.makedirs(step_bitmap_dir)        
        if not os.path.exists(step_svg_dir):
            os.makedirs(step_svg_dir)
        with open(os.path.join(step_svg_dir, str(file_idx)+'.svg'), 'w+') as f:
            f.write(doc.toxml())
        cairosvg.svg2png(url=step_svg_dir+str(file_idx)+'.svg', write_to= step_bitmap_dir+str(file_idx)+'.png')

        image = self.load_image(step_bitmap_dir+str(file_idx)+'.png', transform)
        return image

    def get_original_data(self, element, data):
        recovered_data = [] 

        r_min = 10
        r_max = 200
        svg_width = 500
        svg_height = 500

        # radius = str((int(data[0])* r_max) + (r_max  - r_min)*2)
        # cx  = str(int(data[1])*500 + 250)
        # cy = str(int(data[2])*500 + 250)
        # x = str(int(data[3]))
        # y = str(int(data[4]))
        # width= str(int(data[5]))
        # heigtht = str(int(data[6]))
        # x1 = str(int(data[7]))
        # y1 = str(int(data[8]))
        # x2 = str(int(data[9]))
        # y2 = str(int(data[10]))
        # stroke_width = str(int(data[11]))
        # r = str(int(data[12]*128)+256)
        # g = str(int(data[13]*128)+256)
        # b = str(int(data[14]*128)+256)


        radius = str(int(data[0]))
        cx = str(int(data[1]))
        cy = str(int(data[2]))
        r_c  = str(int(data[3]))
        g_c = str(int(data[4]))
        b_c = str(int(data[5]))
        x = str(int(data[8]))
        y = str(int(data[9]))
        width= str(int(data[6]))
        heigtht = str(int(data[7]))
        r_r = str(int(data[10]))
        g_r  = str(int(data[11]))
        b_r =  str(int(data[12]))
        x1 = str(int(data[13]))
        y1 = str(int(data[14]))
        x2 = str(int(data[15]))
        y2 = str(int(data[16]))
        stroke_width = str(int(data[17]))
        r_l =  str(int(data[18]))
        g_l =  str(int(data[19]))
        b_l = str(int(data[20]))




        recovered_data.extend([radius, cx, cy, r_c, g_c, b_c, width, heigtht, x, y, r_r, g_r, b_r,
         x1, x2, y1, y2, stroke_width, r_l, g_l, b_l])


        return recovered_data





    def gen_svg(self, in_data=None):
        #make svg 
        doc = minidom.Document()
        svg_width = '500'
        svg_height = '500'
        svg = doc.createElement('svg')
        svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
        svg.setAttribute("width", svg_width)
        svg.setAttribute("height", svg_height)
        doc.appendChild(svg)

        # g = doc.createElement('g')
        # g.setAttribute("transform", "translate(0,0)")
        # svg.appendChild(g)
        data = {}    
        svg_arr = []
        svg_arr.append(svg.toxml())

        if in_data == None:
            num_element = random.randrange(1,10)
            svg_elements = ['circle', 'rect', 'line']
            #svg_elements = ['rect']
            element_arr = sorted(np.random.choice(svg_elements, num_element, p=[0.4,0.4, 0.2] ))
            parameters = {}
            parameters['r_min']  = 20 
            parameters['r_max']  = 150
            parameters['length_min'] = 20 
            parameters['length_max'] = 120 
            parameters['svg_height'] = 500
            parameters['svg_width'] = 500 
            parameters['s_width_min'] = 3
            parameters['s_width_max'] = 5
            element_data, label_data, mask = self.gen_svg_element_data(element_arr, parameters)
            out_data = element_data
   
        else: 
            data['svg_elements'] = element_arr

        for i, element in enumerate(element_arr):
            polygon = self.create_element(doc, element, element_data)
            element_data = element_data[21:]
            svg.appendChild(polygon)
            svg_arr.append(svg.toxml())

        return doc, svg_arr, element_arr, label_data, mask

    def create_element(self, doc, element, element_data):

        if element == 'circle':
            polygon = doc.createElement('circle')
            polygon.setAttribute('r', element_data[0])
            polygon.setAttribute('cx', element_data[1])
            polygon.setAttribute('cy', element_data[2])
            color = self.format_color(int(element_data[3]),
                int(element_data[4]), int(element_data[5]))
            polygon.setAttribute('style', 'fill:'+color+';')

        elif element == 'rect':
            polygon = doc.createElement('rect')
            polygon.setAttribute('width', element_data[6])
            polygon.setAttribute('height', element_data[7])
            polygon.setAttribute('x', element_data[8])
            polygon.setAttribute('y', element_data[9])
            color = self.format_color(int(element_data[10]), 
                int(element_data[11]), int(element_data[12]))
            polygon.setAttribute('style','fill:'+color+';')

        elif element == 'line':
            polygon = doc.createElement('line')
            polygon.setAttribute('x1', element_data[13])
            polygon.setAttribute('y1', element_data[14])
            polygon.setAttribute('x2', element_data[15])
            polygon.setAttribute('y2', element_data[16])
            stroke_width = element_data[17]
            color = self.format_color(int(element_data[18]), 
                int(element_data[19]), int(element_data[20]))
            polygon.setAttribute('style', 'stroke:'+color+';stroke-width:'+stroke_width)

        return polygon


    def format_color(self, r,g,b):
        return '#{:02x}{:02x}{:02x}'.format(r,g,b)



    def gen_svg_element_data(self, element_arr, parameters):
        element_data = []
        label_data = []
        tot_data = []
        r_min = parameters['r_min'] 
        r_max = parameters['r_max']  
        length_min = parameters['length_min'] 
        length_max = parameters['length_max'] 
        svg_height = parameters['svg_height'] 
        svg_width = parameters['svg_width'] 
        s_width_min = parameters['s_width_min'] 
        s_width_max = parameters['s_width_max'] 

        mask = np.zeros([svg_height, svg_width, len(element_arr)])

        for i, element in enumerate(element_arr): 

            temp_arr = []

            r = random.randint(0,255)
            g = random.randint(0,225)
            b = random.randint(0,255)

            if element == 'circle':
                radius = random.randrange(r_min, r_max)
                cx = random.randrange(radius, svg_width - radius)
                cy = random.randrange(radius, svg_height - radius)
                element_data.append(str(radius))
                element_data.append(str(cx))
                element_data.append(str(cy))
                element_data.append(str(r))
                element_data.append(str(g))
                element_data.append(str(b))
                element_data.extend(['0', '0', '0', '0', '0', '0', '0'])
                element_data.extend(['0', '0', '0', '0', '0', '0', '0', '0'])  

                xmin = str(cx - radius)
                ymin = str(cy - radius)
                xmax = str(cx + radius)
                ymax = str(cy + radius)  
                mask[:,:,i:i+1] = cv2.circle(mask[:,:,i:i+1].copy(), (cx, cy), radius, (r,g,b), -1)

                temp_arr.extend([str(radius//4 + 1), str(cx//4+1), str(cy//4+1), str(r),str(g), str(b)]) 

            elif element == 'rect':
                width = random.randrange(length_min, length_max)
                height = random.randrange(length_min, length_max)
                x = random.randrange(0, svg_width - width)
                y = random.randrange(0, svg_height - height)
                element_data.extend(['0', '0', '0', '0', '0', '0'])
                element_data.append(str(width))
                element_data.append(str(height))
                element_data.append(str(x))
                element_data.append(str(y))
                element_data.append(str(r))
                element_data.append(str(g))
                element_data.append(str(b))
                element_data.extend(['0', '0', '0', '0', '0','0', '0', '0'])

                xmin = str(x)
                ymin = str(y)
                xmax = str(x + width)
                ymax = str(y + height)
                mask[:,:,i:i+1] = cv2.rectangle(mask[:,:,i:i+1].copy(), 
                    (x, y), (x + width, y + height), (r,g,b), -1)

                temp_arr.extend([str(x//4+1), str(y//4+1), str(width//4+1), str(height//4+1), str(r),str(g), str(b)]) 


            elif element == 'line':
                x_1 = random.randrange(0,svg_width)
                y_1 = random.randrange(0,svg_height)
                x_2 = random.randrange(0,svg_width)
                y_2 = random.randrange(0,svg_height)
                stroke_width = random.randrange(s_width_min,s_width_max)
                element_data.extend(['0', '0', '0', '0', '0', '0'])
                element_data.extend(['0', '0', '0', '0', '0', '0', '0'])
                element_data.append(str(x_1))
                element_data.append(str(y_1))
                element_data.append(str(x_2))
                element_data.append(str(y_2))
                element_data.append(str(stroke_width))
                element_data.append(str(r))
                element_data.append(str(g))
                element_data.append(str(b))
                xmin = str(min(x_1,x_2))
                ymin = str(min(y_1,y_2))
                xmax = str(max(x_1,x_2))
                ymax = str(max(y_1,y_2))
                mask[:,:,i:i+1] = cv2.line(mask[:,:,i:i+1].copy(), (x_1, 
                    y_1), (x_2, y_2), (r,g,b), stroke_width)

                temp_arr.extend([str(x_1//4+1), str(y_1//4+1), str(x_2//4+1), str(y_2//4+1), str(stroke_width//4+1), str(r), str(g), str(b)]) 

            tot_data.extend(temp_arr)

        occlusion = np.logical_not(mask[:, :, -1])
        for i in range(len(element_arr)-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        class_ids = np.array([['BG', 'circle', 'rect', 'line'].index(s) for s in element_arr])
        
        rand_num = random.randint(0, len(tot_data)-1)
        label_data = tot_data[rand_num]

        mask = cv2.resize(mask, (128,128))



        return element_data, tot_data, mask 



def main():

    SVGGEN = SVGGenerator()

    root_path = '../dataset/polygon_n/'
    trg_bitmap_dir = root_path + 'bitmap/'
    trg_svg_dir = root_path + 'svg/'
    trg_label_dir = root_path + 'caption/'
    trg_mask_dir = root_path + 'mask/'



    if not os.path.exists((root_path)):
        os.makedirs((root_path))
    if not os.path.exists(trg_bitmap_dir):
        os.makedirs(trg_bitmap_dir)
    if not os.path.exists(trg_svg_dir):
        os.makedirs(trg_svg_dir)
    if not os.path.exists(trg_label_dir):
        os.makedirs(trg_label_dir)

    for i in range(50000):
        if i%100 == 0:
            print(i) 

        doc, svg_arr, element_arr, element_data, mask = SVGGEN.gen_svg()

        svg_name = str(i) + '.svg'
        bitmap_name = str(i) + '.png'
        with open(os.path.join(trg_svg_dir, svg_name), 'w+') as f:
            f.write(doc.toxml())

        with open(os.path.join(trg_label_dir, svg_name), 'w+') as f:
            label_str = ' '.join(element_data)
            f.write(label_str)


        cairosvg.svg2png(url=trg_svg_dir+str(i)+'.svg', write_to= trg_bitmap_dir+str(i)+'.png')


if __name__ == '__main__':

    main()