# TensorShare

The `TensorShare` schema is the main class of the project. It's used to share tensors between different backends.

This schema inherits from the [`pydantic.BaseModel`](https://docs.pydantic.dev/latest/usage/models/#basic-model-usage) 
class and has two fields:

* `tensors`: a base64 encoded string of the serialized tensors
* `size`: the size of the tensors in bytes

## Creating a `TensorShare` object

After [installing the package](../installation) in your project, the TensorShare class can be imported from the
`tensorshare` module.

```python
from tensorshare import TensorShare

ts = TensorShare(
    tensors=...,  # Base64 encoded tensors to byte strings ready to be sent
    size=...,  # Size of the tensors in pydantic.ByteSize format
)
```

## Serializing tensors - `from_dict`

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
    The backend can be specified as a string or as a `Backend` Enum value. Check the [Backends](../usage/backends) section
    for more information.

```python
import torch
from tensorshare import TensorShare

tensors = {
    "embeddings": torch.zeros((2, 2)),
    "labels": torch.zeros((2, 2)),
}
ts = TensorShare.from_dict(tensors, backend="torch")
print(ts)

>>> tensors=b'gAAAAAAAAAB7ImVt...' size=168
```

If you don't specify the backend, the package will try to infer it from the first tensor in the dictionary, which
isn't always the best optimization. As a general rule, it's better to specify the backend explicitly.

!!! warning
    It's not possible (at the moment) to mix tensors from different backends in the same dictionary.
    The `from_dict` method will raise an exception if you try to do so.

### backend-specific examples

Here are some examples of how to create a `TensorShare` object from a dictionary of tensors in different backends.

=== "Flax"

    ``` python
    import jax.numpy as jnp
    from tensorshare import TensorShare

    tensors = {
        "embeddings": jnp.zeros((2, 2)),
        "labels": jnp.zeros((2, 2)),
    }
    ts = TensorShare.from_dict(tensors, backend="flax")
    ```

=== "NumPy"

    ``` python
    import numpy as np
    from tensorshare import TensorShare

    tensors = {
        "embeddings": np.zeros((2, 2)),
        "labels": np.zeros((2, 2)),
    }
    ts = TensorShare.from_dict(tensors, backend="numpy")
    ```

=== "PaddlePaddle"

    ``` python
    import paddle
    from tensorshare import TensorShare

    tensors = {
        "embeddings": paddle.zeros((2, 2)),
        "labels": paddle.zeros((2, 2)),
    }
    ts = TensorShare.from_dict(tensors, backend="paddlepaddle")
    ```

=== "PyTorch"

    ``` python
    import torch
    from tensorshare import TensorShare

    tensors = {
        "embeddings": torch.zeros((2, 2)),
        "labels": torch.zeros((2, 2)),
    }
    ts = TensorShare.from_dict(tensors, backend="torch")
    ```

=== "TensorFlow"

    ``` python
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

``` python
ts = TensorShare(
    tensors=...,  # Base64 encoded tensors to byte strings ready to be sent
    size=...,  # Size of the tensors in pydantic.ByteSize format
)
tensors = ts.to_tensors(backend=...)
```

!!! tip
    Here again, the backend can be specified as a string or as a `Backend` Enum value.
    Check the [Backends](../usage/backends) section for more information.

Here are some examples of how to deserialize the tensors from a `TensorShare` object in different backends.
You need to have the desired backend installed in your project to be able to deserialize the tensors in it.

=== "Flax"

    ``` python
    from tensorshare import TensorShare

    ts = TensorShare(
        tensors=...,  # Base64 encoded tensors to byte strings ready to be sent
        size=...,  # Size of the tensors in pydantic.ByteSize format
    )
    # Get jaxlib.xla_extension.ArrayImpl tensors
    tensors_flax = ts.to_tensors(backend="flax")  # or backend=Backend.FLAX
    ```

=== "NumPy"

    ``` python
    from tensorshare import TensorShare

    ts = TensorShare(
        tensors=...,  # Base64 encoded tensors to byte strings ready to be sent
        size=...,  # Size of the tensors in pydantic.ByteSize format
    )
    # Get numpy.ndarray tensors
    tensors_numpy = ts.to_tensors(backend="numpy")  # or backend=Backend.NUMPY
    ```

=== "PaddlePaddle"

    ``` python
    from tensorshare import TensorShare

    ts = TensorShare(
        tensors=...,  # Base64 encoded tensors to byte strings ready to be sent
        size=...,  # Size of the tensors in pydantic.ByteSize format
    )
    # Get paddle.Tensor tensors
    tensors_paddle = ts.to_tensors(backend="paddlepaddle")  # or backend=Backend.PADDLEPADDLE
    ```

=== "PyTorch"

    ``` python
    from tensorshare import TensorShare

    ts = TensorShare(
        tensors=...,  # Base64 encoded tensors to byte strings ready to be sent
        size=...,  # Size of the tensors in pydantic.ByteSize format
    )
    # Get torch.Tensor tensors
    tensors_pytorch = ts.to_tensors(backend="torch")  # or backend=Backend.TORCH
    ```

=== "TensorFlow"

    ``` python
    from tensorshare import TensorShare

    ts = TensorShare(
        tensors=...,  # Base64 encoded tensors to byte strings ready to be sent
        size=...,  # Size of the tensors in pydantic.ByteSize format
    )
    # Get tensorflow.Tensor tensors
    tensors_tensorflow = ts.to_tensors(backend="tensorflow")  # or backend=Backend.TENSORFLOW
    ```
