import numpy as np
import os
from xml.dom import minidom
import random 
import math 
import cairosvg
from gen_bitmap_caption_barchart import BarChartGenerator
from gen_bitmap_caption_linechart import LineChartGenerator
from gen_bitmap_caption_piechart import PieChartGenerator

class ChartDataGenerator():

    def __init__(self):
        super(ChartDataGenerator, self).__init__()


def main():

    LINEGEN = LineChartGenerator()
    PIEGEN = PieChartGenerator()
    BARGEN = BarChartGenerator()

    root_path = '../data/chart_test/'
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
        if i%100 == 0:
            print(i) 

        chart_type = random.randrange(0,3)

        if chart_type % 3 == 0:
            doc, caption = LINEGEN.gen_line_chart()
        elif chart_type % 3 == 1:
            doc, caption = PIEGEN.gen_svg_pie_chart()
        elif chart_type % 3 == 2: 
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