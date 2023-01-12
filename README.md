# Feature Dropout: Revisiting the Role of Augmentations in Contrastive Learning

[Alex Tamkin](https://www.alextamkin.com/), [Margalit Glasgow](https://web.stanford.edu/~mglasgow/), Xiluo He, and [Noah Goodman](http://cocolab.stanford.edu/ndg.html)

Paper link: https://arxiv.org/abs/2212.08378

This code is based off of [Viewmaker Networks: Learning Views for Unsupervised Representation Learning](https://github.com/alextamkin/viewmaker).

## 0) Abstract

What role do augmentations play in contrastive learning? Recent work suggests that
good augmentations are label-preserving with respect to a specific downstream task. We
complicate this picture by showing that label-destroying augmentations can be useful in the
foundation model setting, where the goal is to learn diverse, general-purpose representations
for multiple downstream tasks. We perform contrastive learning experiments on a range of
image and audio datasets with multiple downstream tasks (e.g. for digits superimposed on
photographs, predicting the class of one vs. the other). We find that Viewmaker Networks,
a recently proposed model for learning augmentations for contrastive learning, produce
label-destroying augmentations that stochastically destroy features needed for different
downstream tasks. These augmentations are interpretable (e.g. altering shapes, digits, or
letters added to images) and surprisingly often result in better performance compared to
expert-designed augmentations, despite not preserving label information. To support our
empirical results, we theoretically analyze a simple contrastive learning setting with a linear
model. In this setting, label-destroying augmentations are crucial for preventing one set of
features from suppressing the learning of features useful for another downstream task. Our
results highlight the need for analyzing the interaction between multiple downstream tasks
when trying to explain the success of foundation models.

## 1) Install Dependencies

We used the following PyTorch libraries for Python 3.9.4 and CUDA 10.1; you may have to adapt for your own CUDA version:

```console
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other dependencies:
```console
pip install -r requirements.txt
```

## 2) Running experiments

Start by running
```console
source init_env.sh
```

Now, you can run experiments for the different modalities as follows:

```console
python scripts/run_image.py config/image/pretrain_viewmaker_cifar10shapes_simclr.json --gpu-device 0
```

This command runs viewmaker pretraining on the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) image dataset, overlaid with simple shape features, using GPU #0. (If you have a multi-GPU node, you can specify other GPUs.)

The `scripts` directory holds:
- `run_image.py`: for pretraining and running linear evaluation on CIFAR-10
- `run_audio.py`: for pretraining on AudioMNIST and running linear evaluation on a range of transfer datasets

The `config` directory holds configuration files for the different experiments, specifying the hyperparameters from each experiment. The first field in every config file is `exp_base` which specifies the base directory to save experiment outputs, which you should change for your own setup. The `system` field should be changed to indicate pretraining/transfer training and the type of augmentation (ie. viewmaker or expert views). The `dataset` field is located in `data_params` and should be changed to specify the dataset with the appropriate suppressing feature. The boolean `alternate_label` field is also in `data_params,` and indicates whether you are classifying based on the additional feature (eg. shapes, letters, etc.). 

CIFAR10 and MNIST will automatically download, but AudioMNIST must be manually downloaded [here](https://github.com/soerenab/AudioMNIST/tree/master/data). The paths of downloaded datasets can be changed in `src/datasets/root_paths.py`. 

Training curves and other metrics are logged using [wandb.ai](wandb.ai)

## 3) Description of Different Features

The codebase enables pretraining on different hybrid datasets, consisting of a main label (e.g. CIFAR, AudioMNIST) and an alternate label (Shapes, Digits, Letters, and background noise).

The secondary features present are:
- `Shapes`: A square, triangle, or circle (with random color) placed in the middle of the CIFAR image.
- `Digits`: An MNIST digit duplicated 4 times, each placed in a corner of the CIFAR image.
- `Letters`: Four of the same letters of the English alphabet, all the same random color, overlaid on a CIFAR image.
- `Background noise`: A different kind of background noise (e.g. traffic, machinery, cafe) overlaid on an AudioMNIST spectrogram.

Configs to run each of these experiments (with either viewmaker or expert augmentations) can be found in the `config` directory
