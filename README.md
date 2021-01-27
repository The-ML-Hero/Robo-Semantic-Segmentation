# UNet: semantic segmentation with PyTorch


![input and output for a random image in the test dataset](https://raw.githubusercontent.com/The-ML-Hero/Pytorch-Semantic-Segmentation/master/utils/Github_Graph.png)


Customized implementation of the [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch for high definition images with a simple API to use. It's really easy in just 3 lines of code you can a highly accurate working model.

## Usage
**Note : Use Python 3.6 or newer**

### Python Wrapper(Recommended)

```python 
import pytorch_segmentation as ps
net = ps.make_model(n_channels=3,n_classes=1,bilinear=True) # n_channels is the number of input channels(i.e,3 for rgb,bgr,etc.. and 1 is for grayscale images)
ps.train(net=net,epochs=5,batch_size=4,lr=0.0001,img_scale=0.5,val_percent=10.0) # train the network 
```

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

### Prediction

After training your model and saving it to a  *.pth, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg --model model.pth`
It'll be saved to a file called output.png

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

You can specify which model file to use with `--model MODEL.pth`.

### Training via the CLI

```shell script
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 15.0)

```


