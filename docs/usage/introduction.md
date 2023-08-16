The main goal of the TensorShare package is to provide __easy to use and clear interfaces__ for sharing tensors
between different applications connected to a network.

The package is designed to be used in a distributed environment where multiple applications are running on different
machines and need to share tensors between each other. The tensors could be in a different backend (e.g. `numpy` and `torch`)
and the package will help you to share them in a simple and transparent way.

!!! tip
    The project leverages the best open-source tools and integrates them into a single package to provide you
    a fast and reliable way to share your tensors.

## Main Components

There are 3 main components in the TensorShare package:

* [__TensorShare__](../usage/tensorshare) - the main schema for sharing tensors accross the network and using multiple backends (See [Backends](../usage/backends))
* [__TensorShareServer__](../usage/tensorshare_server) - a server configuration schema for defining how tensors could be shared over the network.
* [__TensorShareClient__](../usage/tensorshare_client) - the http client for sending tensors to the server.

## Easy FastAPI Integration

We also provide a set of tools to integrate TensorShare into your FastAPI application in a matter of seconds. See
[FastAPI Integration](../usage/fastapi_integration) for more details.

## Issues and Feature Requests

If you have any issues or feature requests, please feel free to open an issue on the [GitHub repository](https://github.com/chainyo/tensorshare/issues).
