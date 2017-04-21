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

Download the [data](http://www.plant-phenotyping.org/datasets-home) and save to

```bash
> $ROOT/data
```

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


### TODO ###

* patch-based
* post process with CRF

- e2e for kagglefish

-scaling:
    - find min-max of scales from data
    - for each image
        - scale it rand(min,max)
        - take 256,256 patch of that
