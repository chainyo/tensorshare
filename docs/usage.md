# Usage

## Creating a `TensorShare` object

After [installing the package](../installation) in your project, the TensorShare class can be imported from the
`tensorshare` module.

```python
from tensorshare import TensorShare

ts = TensorShare(
    tensors=...,  # Serialized tensors to byte strings ready to be sent
    size=...,  # Size of the tensors in pydantic.ByteSize format
)
```

## Serializing tensors

Because it's tedious to serialize tensors manually, the package provides a `TensorShare.from_dict` method to create
a new object from a dictionary of tensors in any supported backend.

```python
from tensorshare import TensorShare

tensors = {
    "embeddings": ...,  # Tensor
    "labels": ...,  # Tensor
}
ts = TensorShare.from_dict(tensors)
```

### with a specific backend

You can specify the backend to use by passing the `backend` argument to the `from_dict` method.

!!! tip
    The backend can be specified as a string or as a `Backend` Enum value. Check the [Backends](#backends) section
    for more information.

```python
import torch
from tensorshare import TensorShare

tensors = {
    "embeddings": torch.zeros((2, 2)),
    "labels": torch.zeros((2, 2)),
}
ts = TensorShare.from_dict(tensors, backend="torch")
```

If you don't specify the backend, the package will try to infer it from the first tensor in the dictionary, which
isn't always the best optimization. As a general rule, it's better to specify the backend explicitly.

!!! warning
    It's not possible to mix tensors from different backends in the same dictionary.
    The `from_dict` method will raise an exception if you try to do so.

### backend-specific examples

Here are some examples of how to create a `TensorShare` object from a dictionary of tensors in different backends.

##### Flax

```python
import jax.numpy as jnp
from tensorshare import TensorShare

tensors = {
    "embeddings": jnp.zeros((2, 2)),
    "labels": jnp.zeros((2, 2)),
}
ts = TensorShare.from_dict(tensors, backend="flax")
```

##### NumPy

```python
import numpy as np
from tensorshare import TensorShare

tensors = {
    "embeddings": np.zeros((2, 2)),
    "labels": np.zeros((2, 2)),
}
ts = TensorShare.from_dict(tensors, backend="numpy")
```

##### PaddlePaddle

```python
import paddle
from tensorshare import TensorShare

tensors = {
    "embeddings": paddle.zeros((2, 2)),
    "labels": paddle.zeros((2, 2)),
}
ts = TensorShare.from_dict(tensors, backend="paddlepaddle")
```

##### PyTorch

```python
import torch
from tensorshare import TensorShare

tensors = {
    "embeddings": torch.zeros((2, 2)),
    "labels": torch.zeros((2, 2)),
}
ts = TensorShare.from_dict(tensors, backend="torch")
```

##### TensorFlow

```python
import tensorflow as tf
from tensorshare import TensorShare

tensors = {
    "embeddings": tf.zeros((2, 2)),
    "labels": tf.zeros((2, 2)),
}
ts = TensorShare.from_dict(tensors, backend="tensorflow")
```

## Deserializing tensors

Just like the `from_dict` method, the `to_tensors` method can be used to deserialize the serialized tensors
stored in the `TensorShare` object. The method expects a `backend` argument to specify the backend to use.

```python
ts = TensorShare(
    tensors=...,  # Serialized tensors to byte strings ready to be sent
    size=...,  # Size of the tensors in pydantic.ByteSize format
)
tensors = ts.to_tensors(backend="torch")
```

## Backends

The project currently supports the following backends:

* [x] Flax
* [x] NumPy
* [x] PaddlePaddle
* [ ] TensorFlow (coming soon when `v2.14` is released)
* [x] PyTorch

We provide a `Backend` Enum class to help you choose the backend you want to use or for type hinting purposes.

```python
from tensorshare.serialization import Backend

flax_backend = Backend.FLAX
>>> <Backend.FLAX: 'flax'>

numpy_backend = Backend.NUMPY
>>> <Backend.NUMPY: 'numpy'>

paddlepaddle_backend = Backend.PADDLEPADDLE
>>> <Backend.PADDLEPADDLE: 'paddlepaddle'>

tensorflow_backend = Backend.TENSORFLOW
>>> <Backend.TENSORFLOW: 'tensorflow'>

backend = Backend.TORCH
>>> <Backend.TORCH: 'torch'>
```