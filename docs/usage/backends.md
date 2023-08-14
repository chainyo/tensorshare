## Backends

The project currently supports the following backends:

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
