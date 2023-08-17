<h1 align="center">TensorShare</h1>

<div align="center">
	<a  href="https://pypi.org/project/tensorshare" target="_blank">
		<img src="https://img.shields.io/pypi/v/tensorshare.svg" />
	</a>
	<a  href="https://pypi.org/project/tensorshare" target="_blank">
		<img src="https://img.shields.io/pypi/pyversions/tensorshare" />
	</a>
	<a  href="https://github.com/chainyo/tensorshare/blob/main/LICENSE" target="_blank">
		<img src="https://img.shields.io/pypi/l/tensorshare" />
	</a>
	<a  href="https://github.com/chainyo/tensorshare/actions?workflow=ci-cd" target="_blank">
		<img src="https://github.com/chainyo/tensorshare/workflows/ci-cd/badge.svg" />
	</a>
	<a href="https://codecov.io/gh/chainyo/tensorshare" > 
		<img src="https://codecov.io/gh/chainyo/tensorshare/branch/main/graph/badge.svg?token=IA2W48WCCN"/> 
	</a>
	<a  href="https://github.com/pypa/hatch" target="_blank">
		<img src="https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg" />
	</a>
</div>

<p align="center"><em>🤝 Trade any tensors over the network</em></p>

---

TensorShare is a powerful tool enabling ⚡ lightning-fast tensor sharing across networks.

This project leverages the best open-source tools to provide a simple and easy-to-use interface for sharing tensors:

* [safetensors](https://github.com/huggingface/safetensors) for secure tensor serialization and deserialization.
* [pydantic](https://github.com/pydantic/pydantic) for data validation and settings management.
* [fastapi](https://github.com/tiangolo/fastapi) for building APIs (and because it's too good to avoid it).

__This project is heavily in development and is not ready for production use.__
__Feel free to contribute to the project by opening issues and pull requests.__

## Usage

Example of tensors serialization with torch:

```python
import torch
from tensorshare import TensorShare

tensors = {
    "embeddings": torch.zeros((2, 2)),
    "labels": torch.zeros((2, 2)),
}
ts = TensorShare.from_dict(tensors, backend="torch")
print(ts)
# tensors=b'gAAAAAAAAAB7ImVtYmVkZGluZ3MiOnsiZHR5cGUiO...' size=168
```

You can now freely send the tensors over the network via any means (e.g. HTTP, gRPC, ...).

On the other side, when you receive the tensors, you can deserialize them in any supported backend:

```python
from tensorshare import Backend

np_tensors = ts.to_tensors(backend=Backend.NUMPY)
print(np_tensors)
# {
# 	'embeddings': array([[0., 0.], [0., 0.]], dtype=float32),
# 	'labels': array([[0., 0.], [0., 0.]], dtype=float32)
# }
```

For more examples and details, please refer to the [Usage section](./usage/introduction).

## Roadmap

- [x] Pydantic schema for sharing tensors ~> [TensorShare](./usage/tensorshare.md)
- [x] Serialization and deserialization ~> [tensorshare.serialization](./api/serialization)
- [x] TensorProcessor for processing tensors ~> [TensorProcessor](./api/processor)
- [x] Server for sharing tensors ~> [TensorShareServer](./usage/tensorshare_server.md)
- [x] Client for sharing tensors ~> [TensorShareClient](./usage/tensorshare_client.md)
- [ ] New incredible features! (Check the [issues](https://github.com/chainyo/tensorshare/issues))
