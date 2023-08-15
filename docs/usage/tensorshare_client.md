The `TensorShareClient` class is a core component of this project, enabling users to send tensors to a remote server 
using asynchronous HTTP requests.

It's built on top of the [`aiohttp`](https://github.com/aio-libs/aiohttp) library, which allows for asynchronous HTTP communication, enhancing the efficiency and speed of tensor sharing.

This class extends the TensorShare ecosystem by providing both synchronous and asynchronous interfaces for 
sending tensors, ensuring flexibility and adaptability to various use cases and environments.

!!! note
    While it's inherently asynchronous (thanks to the `aiohttp`` library), the class provides 
    a synchronous interface for users who prefer it, especially for testing and non-production use.
    
    However, due to potential performance implications, using the synchronous method in production environments 
    isn't recommended.

## Pre-requisites

This module requires the `aiohttp` library to be installed:

``` bash
pip install tensorshare[client]
```

## Initialization

To create an instance of `TensorShareClient`, users need to pass in the server configuration, a timeout duration
(defaulted to 10 seconds), and optionally decide if they want to validate server endpoints at initialization.

``` python
from tensorshare import TensorShareServer, TensorShareClient

server_config = TensorShareServer.from_dict(
    server_config={"url": "http://localhost:5000"}
)
client = TensorShareClient(server_config, timeout=10, validate_endpoints=True)
```

## Parameters

* __server_config__ ([`TensorShareServer`](../usage/tensorshare_server)): The configuration for the remote server.
* __timeout__ (`int`): Specifies the maximum duration (in seconds) to wait for an HTTP response. Defaults to 10 seconds.
* __validate_endpoints__ (`bool`): Determines whether the client should validate the server's endpoints upon initialization. Default is True (recommended).

## Methods

The methods are available in both synchronous and asynchronous forms, with the latter being prefixed with `async_`.
The asynchronous methods are recommended for production use, as they are non-blocking and more efficient.

!!! note
    The synchronous methods are using the asynchronous ones under the hood wrapped in a `asyncio.run` call,

### `ping_server`

The `ping_server` method sends a `GET` request to the server's `/ping` endpoint, which returns a `DefaultResponse`
object with a `status_code` of `200` if the server is up and running.

=== "Sync"

    ``` python
    from tensorshare import TensorShareServer, TensorShareClient

    server_config = TensorShareServer.from_dict(
        server_config={"url": "http://localhost:5000"}
    )
    client = TensorShareClient(server_config)

    r = client.ping_server()
    print(r)
    >>> True
    ```

=== "Async"

    ``` python
    import asyncio
    from tensorshare import TensorShareServer, TensorShareClient

    server_config = TensorShareServer.from_dict(
        server_config={"url": "http://localhost:5000"}
    )
    client = TensorShareClient(server_config)

    async def main():
        r = await client.async_ping_server()
        print(r)

    asyncio.run(main())
    >>> True
    ```

### `send_tensor`

The `send_tensor` method sends a tensor to the server's `/receive_tensor` endpoint, which returns the `response_model`
defined in the server's configuration.

=== "Sync"

    ``` python
    import torch
    from tensorshare import TensorShare, TensorShareServer, TensorShareClient

    server_config = TensorShareServer.from_dict(
        server_config={"url": "http://localhost:5000"}
    )
    client = TensorShareClient(server_config)

    ts = TensorShare.from_dict({"embeddings": torch.rand(10, 10)})

    r = client.send_tensor(ts)
    ```

=== "Async"

    ``` python
    import asyncio
    import torch
    from tensorshare import TensorShare, TensorShareServer, TensorShareClient

    server_config = TensorShareServer.from_dict(
        server_config={"url": "http://localhost:5000"}
    )
    client = TensorShareClient(server_config)

    ts = TensorShare.from_dict({"embeddings": torch.rand(10, 10)})

    async def main():
        r = await client.async_send_tensor(ts)

    asyncio.run(main())
    ```
