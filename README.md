### Learning to count leaves ###
Torch7 code for training a CNN for semantic and instance segmentation of leaves. 

### Directory structure ###

Directory    | Content 
:-------------:|:----------------------
CNN     | Core CNN code
data     | Training data
shells      | various bash shells

### Dependencies ###

install dependencies

```bash
> shells/dependencies.sh
```

### Data ###

We used [Plant Phenotyping DataSet](https://www.plant-phenotyping.org/datasets-overview) for training. 
[Download](http://www.plant-phenotyping.org/datasets-home) the dataset and copy to

```bash
> $ROOT/data
```
see [dataset](CNN/opts.lua) CLI argument for adjusting the path to desired location. 

### Training ###

See [opts.lua](CNN/opts.lua) for CLI options.

```bash
> th -i main.lua
```

### Inference ###

See [inference.lua](CNN/inference.lua) for CLI options.

```bash
> th -i inference.lua
```

### Related (CNN-based) Efforts ###

[DEEP-PLANT: PLANT IDENTIFICATION WITH CONVOLUTIONAL NEURAL NETWORKS](https://arxiv.org/pdf/1506.08425.pdf)

[Plant recognition using CNNs](http://llcao.net/cu-deeplearning15/project_final/Plant%20Recognition.pdf)

[Fine-tuning Deep Convolutional Networks for Plant Recognition](http://ceur-ws.org/Vol-1391/121-CR.pdf)

[Resnet training in Torch](https://github.com/facebook/fb.resnet.torch)

[Faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn)

[Instance-aware Semantic Segmentation via Multi-task Network Cascades](https://github.com/daijifeng001/MNC)

[Recurrent Instance Segmentation (Does Leaf counting)](http://www.robots.ox.ac.uk/~tvg/publications/2016/RIS7.pdf)

[code for RIS (above)](https://github.com/bernard24/ris)

[Paper that inspired RIS](https://arxiv.org/pdf/1506.04878v3.pdf)

[Deepmask](https://github.com/facebookresearch/deepmask)

[Berkely Fully Convolutional Networks](https://github.com/shelhamer/fcn.berkeleyvision.org)

[Leafsnap: A Computer Vision System for Automatic Plant Species Identification](http://neerajkumar.org/base/papers/nk_eccv2012_leafsnap.pdf)  [code](https://github.com/sujithv28/Deep-Leafsnap)



### TODO ###

* patch-based
* post process with CRF

- e2e for kagglefish

-scaling:
    - find min-max of scales from data
    - for each image
        - scale it rand(min,max)
        - take 256,256 patch of that
