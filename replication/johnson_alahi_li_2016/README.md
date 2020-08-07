# Images

```sh
python images.py --help
```

```
usage: download_images.py [-h] [--images-source-dir IMAGES_SOURCE_DIR]
                          [--dataset-dir DATASET_DIR] [--no-dataset]

Download the images for the paper 'Perceptual losses for real-time style
transfer and super-resolution' by Johnson, Alahi, and Li.

optional arguments:
  -h, --help            show this help message and exit
  --images-source-dir IMAGES_SOURCE_DIR
                        Images source directory. Defaults to
                        data/images/source relative to this script.
  --dataset-dir DATASET_DIR
                        Dataset directory. Defaults to data/dataset relative
                        to this script.
  --no-dataset          If given, do not download the dataset (~13GB).
```

# Training

```sh
python training.py --help
```

```
usage: training.py [-h] [--images-source-dir IMAGES_SOURCE_DIR]
                   [--models-dir MODELS_DIR] [--dataset-dir DATASET_DIR]
                   [--no-impl-params] [--no-instance-norm] [--device DEVICE]
                   [--quiet]
                   [style [style ...]]

Training for the paper 'Perceptual losses for real-time style transfer and
super-resolution' by Johnson, Alahi, and Li.

positional arguments:
  style                 Style images for which the training is performed
                        successively. If relative path, the image is searched
                        in IMAGES_SOURCE_DIR. Can also be a valid key from the
                        built-in images. Defaults to all built-in style
                        images.

optional arguments:
  -h, --help            show this help message and exit
  --images-source-dir IMAGES_SOURCE_DIR
                        Images source directory. Defaults to
                        data/images/source relative to this script.
  --models-dir MODELS_DIR
                        Models directory. Defaults to data/models relative to
                        this script.
  --dataset-dir DATASET_DIR
                        Dataset directory. Defaults to data/dataset relative
                        to this script.
  --no-impl-params      If given, use the parameters reported in the paper
                        rather than the ones used in the implementation.
  --no-instance-norm    If given, use batch rather than instance
                        normalization.
  --device DEVICE       Device the training is performed on. Defaults to the
                        available hardware preferring CUDA over CPU.
  --quiet               Do not print training information to STDOUT.
```

# Stylization

```sh
python stylization.py --help
```

```
usage: stylization.py [-h] [--images-source-dir IMAGES_SOURCE_DIR]
                      [--images-results-dir IMAGES_RESULTS_DIR]
                      [--models-dir MODELS_DIR] [--no-impl-params]
                      [--no-instance-norm] [--device DEVICE]
                      style [content [content ...]]

Stylization for the paper 'Perceptual losses for real-time style transfer and
super-resolution' by Johnson, Alahi, and Li.

positional arguments:
  style                 Style the transformer was trained on.
  content               Content images for which the stylization is performed
                        successively. If relative path, the image is searched
                        in IMAGES_SOURCE_DIR. Can also be a valid key from the
                        built-in images. Defaults to all built-in content
                        images.

optional arguments:
  -h, --help            show this help message and exit
  --images-source-dir IMAGES_SOURCE_DIR
                        Images source directory. Defaults to
                        data/images/source relative to this script.
  --images-results-dir IMAGES_RESULTS_DIR
                        Images results directory. Defaults to
                        data/images/results relative to this script.
  --models-dir MODELS_DIR
                        Models directory. Defaults to data/models relative to
                        this script.
  --no-impl-params      If given, use the parameters reported in the paper
                        rather than the ones used in the implementation.
  --no-instance-norm    If given, use batch rather than instance
                        normalization.
  --device DEVICE       Device the training is performed on. Defaults to the
                        available hardware preferring CUDA over CPU.
```

# `LuaTorch` to `PyTorch`

Download and convert the weights provided by Johnson, Alahi, and Li from LuaTorch to 
PyTorch.

```sh
python luatorch_to_pytorch.py
```

The conversion requires `torch < 1`. You can use `ltt install` with the `--force-cpu` 
for a minimal installation:

```sh
pip install light-the-torch
ltt install --force-cpu numpy "torch<1"
```
