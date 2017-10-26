import os 
from xml.dom import minidom

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
    
    radius = radius.split('.')[0]
    style = style.replace("fill: hsl(","").split('.')[0]
    
    
    attr_list.append(pos_x)
    attr_list.append(pos_y)
    attr_list.append(shape)
    attr_list.append(radius)
    attr_list.append(style)
    
    tot_list.append(attr_list)
    
    with open(os.path.join(newpath, f_list), 'w+') as f:
        attr_str = ' '.join(attr_list)
        f.write(attr_str)
