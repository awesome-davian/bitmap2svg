import os 
from xml.dom import minidom
import argparse



def pos_classifier(pos_x, pos_y):
    
    pos_x = int(pos_x)
    pos_y = int(pos_y)   
    sector = 0 
    
    if pos_x >= 0 and pos_x <= 166 and pos_y >=0 and pos_y <= 166:
        sector = 1 
    elif pos_x > 166 and pos_x <= 332 and pos_y >=0 and pos_y <= 166:
        sector = 2
    elif pos_x > 332 and pos_x <= 500 and pos_y >=0 and pos_y <= 166:
        sector = 3
    elif pos_x >= 0 and pos_x <= 166 and pos_y >166 and pos_y <= 332:
        sector = 4
    elif pos_x > 166 and pos_x <= 332 and pos_y >166 and pos_y <= 332:
        sector = 5
    elif pos_x > 332 and pos_x <= 500 and pos_y >166 and pos_y <= 332:
        sector = 6
    elif pos_x >= 0 and pos_x <= 166 and pos_y >332 and pos_y <= 500:
        sector = 7
    elif pos_x > 166 and pos_x <= 332 and pos_y >332 and pos_y <= 500:
        sector = 8
    elif pos_x > 332 and pos_x <= 500 and pos_y >332 and pos_y <= 500:
        sector = 9
    
    return str(sector)
  
def parse_attribute(pos, polygon, attr_list):

    shape = polygon.getAttribute('class')
    attr_list.append(shape)
    
    # calculate sector 
    pos = pos.replace("translate(", "")
    pos = pos.replace(")", "")
    pos_x = pos.split(',')[0]
    pos_y = pos.split(',')[1]     
    sector = pos_classifier(pos_x, pos_y)
    attr_list.append(sector)

    if shape == 'circle':
        radius = polygon.getAttribute('r')
        radius = radius.split('.')[0]
        radius = (str(round(float(radius)/10)*10))
        attr_list.append(radius)
    elif shape == 'rect':
        width = polygon.getAttribute('width')
        height = polygon.getAttribute('height')
        width = (str(round(float(width)/10)*10))
        height = (str(round(float(height)/10)*10))
        attr_list.append(width)
        attr_list.append(height)


    style = polygon.getAttribute('style') 
    style = style.replace("fill: hsl(","").split('.')[0]       
    #attr_list.append(style)

    #print(attr_list)
    
    return attr_list

def parse_attribute_type2(polygon, attr_list):

    shape = polygon.getAttribute('class') 
    attr_list.append(shape)
  
    
    if shape == 'circle':
        radius = polygon.getAttribute('r')
        radius = radius.split('.')[0]
        radius = (str(round(float(radius)/10)*10))  
        
        cx = polygon.getAttribute('cx')
        cy = polygon.getAttribute('cy')
        sector= pos_classifier(cx,cy)
        
        attr_list.append(sector)
        attr_list.append(radius)
        
    elif shape == 'rect':
        x = polygon.getAttribute("x")
        y = polygon.getAttribute("y")
        sector = pos_classifier(x,y) 
        attr_list.append(sector)
        
        width = polygon.getAttribute('width')
        height = polygon.getAttribute('height')
        width = (str(round(float(width)/10)*10))
        height = (str(round(float(height)/10)*10))
        
        attr_list.append(width)
        attr_list.append(height)


    style = polygon.getAttribute('style') 
    style = style.replace("fill: hsl(","").split('.')[0]       
    #attr_list.append(style)

    #print(attr_list)
    
    return attr_list
    



def main(args):

    svg_path  = args.svg_path
    caption_path = args.caption_path
    # Create model directory
    if not os.path.exists(args.caption_path):
        os.makedirs(args.caption_path)

    file_list = os.listdir(svg_path)

    for f_list in file_list:

        fname = os.path.join(svg_path, f_list)
        doc = minidom.parse(fname)
        svg = (doc.getElementsByTagName('svg'))    
        attr_list = []
        
        if svg[0].firstChild.tagName == 'g':
            g = svg[0].firstChild
            polygon = g.firstChild
            pos = g.getAttribute('transform')
            attr_list = parse_attribute(pos, polygon, attr_list)
       
        else:
            for node in svg[0].childNodes:
                if node.hasAttribute('transform') : 
                    pos = node.getAttribute('transform')
                    attr_list = parse_attribute(pos, node, attr_list)
                else:
                    attr_list = parse_attribute_type2(node, attr_list)

                
        with open(os.path.join(caption_path, f_list), 'w+') as f:
            attr_str = ' '.join(attr_list)
            f.write(attr_str)
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/3object/caption', 
                        help='path for train annotation file')

    parser.add_argument('--svg_path', type=str, 
                        default='data/3object/svg', 
                        help='ath for train annotation file')

    args = parser.parse_args()
    main(args)