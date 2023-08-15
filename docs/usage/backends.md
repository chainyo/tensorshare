## Supported backends

The project currently supports the following ML backends:

* [x] Flax
* [x] NumPy
* [x] PaddlePaddle
* [ ] TensorFlow (coming soon when `v2.14` is released)
* [x] PyTorch

#### Backend Enum

We provide a `Backend` Enum class to help you choose the backend you want to use or for type hinting purposes.

```python
from tensorshare import Backend

flax_backend = Backend.FLAX
>>> <Backend.FLAX: "flax">

numpy_backend = Backend.NUMPY
>>> <Backend.NUMPY: "numpy">

paddlepaddle_backend = Backend.PADDLEPADDLE
>>> <Backend.PADDLEPADDLE: "paddlepaddle">

tensorflow_backend = Backend.TENSORFLOW
>>> <Backend.TENSORFLOW: "tensorflow">

backend = Backend.TORCH
>>> <Backend.TORCH: "torch">
```

!!! tip
    When a method is requiring you to specify a `backend` you can choose to use the `Backend` Enum class or the
    string representation of the backend. For example, the following two lines are equivalent:

    ```python
    ts = TensorShare(...)

    tensors = ts.to_tensors(backend=Backend.FLAX)
    tensors = ts.to_tensors(backend="flax")
    ```

    One is prone to typos, the other enables type hinting and IDE autocompletion.

    The choice is yours. 😉

#### TensorType Enum

We also provide a `TensorType` Enum class to help you choose the tensor type you want to use or for type hinting
purposes.

```python
from tensorshare import TensorType

flax_tensor = TensorType.FLAX
>>> <TensorType.FLAX: "jaxlib.xla_extension.ArrayImpl">

numpy_tensor = TensorType.NUMPY
>>> <TensorType.NUMPY: "numpy.ndarray">

paddlepaddle_tensor = TensorType.PADDLEPADDLE
>>> <TensorType.PADDLEPADDLE: "paddle.Tensor">

tensorflow_tensor = TensorType.TENSORFLOW
>>> <TensorType.TENSORFLOW: "tensorflow.Tensor">

torch_tensor = TensorType.TORCH
>>> <TensorType.TORCH: "torch.Tensor">
```
