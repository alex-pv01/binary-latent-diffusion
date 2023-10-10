# Binary Latent Diffusion model
This repo pretends to be an implementation of the Binary Latent Diffusion model. [Paper](https://arxiv.org/pdf/2304.04820.pdf) available in arxiv.

PLEASE BE PATIENT, THIS IS STILL A WORK IN PROGRESS.

## Train Binary Autoencoder
```
python main.py --base configs/bvae.yaml -t True --gpus 0,1
```

### Resume training
```
python main.py --base configs/bvae.yaml --resume PATH/TO/last.ckpt -t True --gpus 0,1
```

## Get data paths
```
find $(pwd)/datasets/coco/train2017 -name "*.jpg" > train.txt
```