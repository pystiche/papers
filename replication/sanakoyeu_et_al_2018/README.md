# `Tensorflow` to `PyTorch`

Download and convert the weights provided by Sanakoyeu et al. from Tensorflow to 
PyTorch.

```sh
python tensorflow_to_pytorch.py
```

The conversion requires `pystiche_papers` and additionally `tensorflow`. You can use 
`light-the-torch` with the `--force-cpu` flag for a minimal installation:

```sh
pip install light-the-torch
ltt install --force-cpu pystiche_papers tensorflow
```
