import os 
from xml.dom import minidom



def pos_classifier(pos_x, pos_y):
    
    pos_x = int(pos_x)
    pos_y = int(pos_y)   
    sector = 0 
    
    if pos_x > 0 and pos_x <= 166 and pos_y >0 and pos_y <= 166:
        sector = 1 
    elif pos_x > 166 and pos_x <= 332 and pos_y >0 and pos_y <= 166:
        sector = 2
    elif pos_x > 332 and pos_x <= 500 and pos_y >0 and pos_y <= 166:
        sector = 3
    elif pos_x > 0 and pos_x <= 166 and pos_y >166 and pos_y <= 332:
        sector = 4
    elif pos_x > 166 and pos_x <= 332 and pos_y >166 and pos_y <= 332:
        sector = 5
    elif pos_x > 332 and pos_x <= 500 and pos_y >166 and pos_y <= 332:
        sector = 6
    elif pos_x > 0 and pos_x <= 166 and pos_y >332 and pos_y <= 500:
        sector = 7
    elif pos_x > 166 and pos_x <= 332 and pos_y >332 and pos_y <= 500:
        sector = 8
    elif pos_x > 332 and pos_x <= 500 and pos_y >332 and pos_y <= 500:
        sector = 9
    
    return str(sector)
  

path  = 'bitmap2svg_samples/svg'
newpath ='bitmap2svg_samples/caption'
file_list = os.listdir(path)

tot_list = []
for f_list in file_list:
    fname = os.path.join(path, f_list)
    doc = minidom.parse(fname)
    svg = (doc.getElementsByTagName('svg'))
    g = svg[0].firstChild
    polygon = g.firstChild
    
    attr_list = []
    
    pos = g.getAttribute('transform')
    shape = polygon.getAttribute('class')
    radius = polygon.getAttribute('r')
    style = polygon.getAttribute('style')
    
    pos = pos.replace("translate(", "")
    pos = pos.replace(")", "")
    pos_x = pos.split(',')[0]
    pos_y = pos.split(',')[1]
    
    sector = pos_classifier(pos_x, pos_y)
    
    radius = radius.split('.')[0]
    style = style.replace("fill: hsl(","").split('.')[0]
    
    
    attr_list.append(sector)
    attr_list.append(shape)
    attr_list.append(radius)
    attr_list.append(style)
    
    tot_list.append(attr_list)
    
    with open(os.path.join(newpath, f_list), 'w+') as f:
        attr_str = ' '.join(attr_list)
        f.write(attr_str)