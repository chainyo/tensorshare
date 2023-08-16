# Installation

- Python `3.8`, `3.9`, `3.10`, or `3.11` is required.

##### Pip

```bash
pip install tensorshare
```

##### Poetry
```bash
poetry add tensorshare
```

##### Pipx

```bash
pipx install tensorshare
```

## TensorShare modules

TensorShare is a modular library. It means that you can install only the modules you need to reduce the installation
time and the number of dependencies.

!!! note
    We do care about your CI/CD pipelines. That's why we provide a way to install only the modules you need.
    Only `safetensors` and `pydantic` are mandatory.

### Client module

The client module is used to create a [`TensorShareClient`](../usage/tensorshare_client) for sending tensors to a
FastAPI server.

```bash
pip install tensorshare[client]
```

!!! note
    It just installs [`aiohttp`](https://github.com/aio-libs/aiohttp) on top of the main dependencies.

### Server module

The server module is used to integrate TensorShare with a FastAPI server. 

```bash
pip install tensorshare[server]
```

!!! note
    It just installs [`fastapi`](https://github.com/tiangolo/fastapi) on top of the main dependencies.

Check the [TensorShareServer](../usage/tensorshare_server) and the [FastAPI integration](../usage/fastapi_integration) sections for more details.

## Backend Installation

TensorShare is a framework-agnostic library. It means that the default installation does not include any framework
and assumes that you will handle the backend (or backends) yourself.

However, we provide a set of backends that can be installed alongside TensorShare.

* [x] [Jax](https://jax.readthedocs.io/en/latest/) | `flax>=0.6.3`, `jax>=0.3.25`, `jaxlib>=0.3.25`

```
pip install tensorshare[jax]
```

* [x] [NumPy](https://numpy.org/) | `numpy>=1.21.6`

```
pip install tensorshare[numpy]
```

* [x] [PaddlePaddle](https://www.paddlepaddle.org.cn/) | `paddlepaddle>=2.4.1`

```
pip install tensorshare[paddlepaddle]
```

* [ ] [Tensorflow](https://www.tensorflow.org/) | `tensorflow>=2.14`

```
pip install tensorshare[tensorflow]
```

* [x] [PyTorch](https://pytorch.org/) | `torch>=1.10`

```
pip install tensorshare[torch]
```

You can also install all the backends at once:

```bash
pip install tensorshare[all]
```

## Development Installation

We use [Hatch](https://github.com/pypa/hatch) as package manager for development.

For installing the default environment:

```bash
hatch env create 
```

The `quality` and `tests` environments are also available. 
They will be auto-activated when running the corresponding commands.

```bash
hatch run quality:format 
hatch run test:run
```

But you can also create them manually:

```bash
hatch env create quality
hatch env create tests
```
