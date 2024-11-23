# CS59300CVD Assignment 2

## Requirements
```python
matplotlib==3.8.4
numpy==2.1.1
Pillow==10.4.0
skimage==0.0
torch==2.4.0
torchvision==0.19.0
```

## How to run

## Part 1

### run all images with each metrics
```python
python main_p1.py -i all
```

### run specific image with specific metric
```python
python main_p1.py -i [image name] -m [metric]
ex) python main_p1.py -i 1 -m ncc
```
- image name: 1 to 6
- metric: {mse, ncc}


Results are saved in `data/part1`.
## Part 2

### run all with default settings
```python
python main_p2.py -i all
```

### run specific image with varying options
```python
python main_p2.py -i [image name] -s [sigma] -k [kernel size] -t [threshold] -n [number of iterations]
```
- image name: {butterfly, einstein, fishes, sunflowers}

Results are saved in `outputs`.