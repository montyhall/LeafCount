### Learning to count leaves ###
Learns semantic and instance segmentation of leaves

Directory    | Content 
:-------------:|:----------------------
CNN     | Core CNN code
data     | Training data
shells      | various bash shells

### Run ###

install dependencies

```bash
> shells/dependencies.sh
```

Get Data

Download the [data](http://www.plant-phenotyping.org/datasets-home) and save to

```bash
> $ROOT/data
```

if running on CIMS (CUDA2) do
```bash
> th -i main.lua
```


Train
```bash
> 
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
