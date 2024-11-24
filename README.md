# Blob Detection
Blob detection is a task that identifies and localizes regions in an image which stand out in terms of intensity, texture, or other properties, often representing features or objects of interest. To detect the blobs, various methods such as Difference of Gaussians (DoG) or Laplacian of Gaussian (LoG) are used to identify regions with significant intensity changes or distinct shapes across different scales.

![image](https://github.com/user-attachments/assets/80bb608a-63af-4902-8fb1-333ab869a5ec)



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
