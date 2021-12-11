# Introduction

The model training procedure implemented here, follows the recipe described in [this blog post on the pytorch official website](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/), and uses part of its code with slight or no modification.

The training recipe are embedded inside hyperparameter schedulers using the [Tune package from Ray](https://docs.ray.io/en/latest/tune/index.html).

## Installation

In order to run the scripts inside this repo, you need a working python environment (preferably anaconda), and to install the following:

```bash
#!/bin/bash
pip install torch, torchvision, ray[tune], timm
pip install git+https://github.com/etetteh/torchxrayvision.git
```
