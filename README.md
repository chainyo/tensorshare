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

This project leverage the best open-source tools to provide a simple and easy-to-use interface for sharing tensors:

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
```

For more examples and details, please refer to the [documentation](https://chainyo.github.io/tensorshare/usage/).

## Roadmap

- [x] Serialization and deserialization
- [x] TensorProcessor for processing tensors
- [x] Pydantic schema for sharing tensors
- [ ] Server for sharing tensors
- [ ] Client for sharing tensors
