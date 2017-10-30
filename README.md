# Bitmap2SVG 

## Get Data
download to './data/' directory
https://drive.google.com/file/d/0B1iIQMDjpzKRcVFONElrNmJHd2c/view?usp=sharing

## Usage
### generate captioning 
```sh
$python svg_caption_gen.py --caption_path 'custom_path' --svg_path 'custom_path' 
```
or simply 
```sh
$python svg_caption_gen.py 
```

### Train model 
```sh
$python main.py 
```
train accuracy : Test_Train_accuracy.ipynb
### Test
```sh
$python sample.py --imager 'data/bitmap2_test/bitmap/image_id.png'
```
Test_Accuracy.ipynb 

