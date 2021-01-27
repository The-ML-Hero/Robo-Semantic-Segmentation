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

### Training via the CLI (Deprecated)

```shell script
> python train.py -h
```


