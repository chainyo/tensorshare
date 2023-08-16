TensorShare provides built-in integration capabilities with [`FastAPI`](https://github.com/tiangolo/fastapi),
a high-performance web framework for building APIs with Python.

By combining the power of TensorShare's tensor sharing capabilities with FastAPI's intuitive and robust API development,
you can easily set up an efficient server-side endpoint to handle tensor computations.

!!! tip
    It's also because not using FastAPI in a project that leverages the best of
    open-source is a crime. Don't be a criminal.

## Pre-requisites

This module obviously needs `fastapi` to be installed:

``` bash
pip install tensorshare[server]
```

## The pre-built router

TensorShare provides a pre-built router that you can use to quickly integrate tensor sharing into any existing
FastAPI application.

### Importing the router

To import the router, simply use the following import statement:

=== "Sync"

    ``` python
    from tensorshare import create_tensorshare_router
    ```

=== "Async"

    ``` python
    from tensorshare import create_async_tensorshare_router
    ```

### Adding it to FastAPI

To add the router to your FastAPI application, simply add it to the `include` parameter of the `APIRouter` constructor:

=== "Sync"

    ``` python
    from fastapi import FastAPI, APIRouter
    from tensorshare import create_tensorshare_router

    app = FastAPI()  # Your FastAPI application

    ts_router = create_tensorshare_router()
    app.include_router(ts_router)
    ```

=== "Async"

    ``` python
    from fastapi import FastAPI, APIRouter
    from tensorshare import create_async_tensorshare_router

    app = FastAPI()  # Your FastAPI application

    ts_router = create_async_tensorshare_router()
    app.include_router(ts_router)
    ```

You can add extra `prefix` or `tags` parameter to the `include_router` method to customize the router:

``` python
app.include_router(ts_router, prefix="/tensorshare", tags=["tensorshare", "tensors"])
```

### Customize the router

If you want to customize the router, you will have to pass two parameters to the `create_tensorshare_router` function:

* __server_config__: A Python dictionary containing the configuration of the server, just like the one you would pass to
    the [`TensorShareServer`](../usage/tensorshare_server#create-a-config---from_dict) constructor.
* __custom_operation__: The custom computation operation to use for the `/receive_tensor` endpoint. You will learn more
    about this in the [Customizing the computation operation](#customizing-the-computation-operation) section.

=== "Sync"

    ``` python
    from fastapi import FastAPI, APIRouter
    from tensorshare import create_tensorshare_router

    class

    app = FastAPI()  # Your FastAPI application

    server_config = {"url": "http://localhost:8000"}

    ts_router = create_tensorshare_router(server_config=server_config)
    app.include_router(ts_router)
    ```

=== "Async"

    ``` python
    from fastapi import FastAPI, APIRouter
    from tensorshare import create_async_tensorshare_router

    app = FastAPI()  # Your FastAPI application

    server_config = {"url": "http://localhost:8000"}

    ts_router = create_async_tensorshare_router(server_config=server_config)
    app.include_router(ts_router)
    ```

### The `ping` endpoint

The router comes with a built-in `ping` endpoint that you can use to check if the server is up and running.

=== "Sync"

    ``` python
    @router.get(
        f"{config.ping.path}",
        response_model=DefaultResponse,
        status_code=http_status.HTTP_200_OK,
        tags=["tensorshare"],
    )
    def ping() -> DefaultResponse:
        """Synchronous endpoint to check if the server is up."""
        return DefaultResponse(message="The TensorShare router is up and running!")
    ```

=== "Async"

    ``` python
        @router.post(
        f"{config.receive_tensor.path}",
        response_model=config.response_model,
        status_code=http_status.HTTP_200_OK,
        tags=["tensorshare"],
    )
    def receive_tensor(
        shared_tensor: TensorShare,
        operation: Callable = Depends(get_computation_operation),
    ) -> Any:
        """Synchronous endpoint to handle tensors reception and computation."""
        result = operation(shared_tensor)

        return result
    ```

There is nothing too fancy here, just a simple endpoint that returns a `DefaultResponse` object with a message.

The `config.ping.path` will be the path defined in the [`TensorShareServer`](../usage/tensorshare_server)
configuration object you provided when creating the router.

### The `receive_tensor` endpoint

The router also comes with a built-in `receive_tensor` endpoint that you can use to receive a tensor from a client.

=== "Sync"

    ``` python
    @router.post(
        f"{config.receive_tensor.path}",
        response_model=config.response_model,
        status_code=http_status.HTTP_200_OK,
        tags=["tensorshare"],
    )
    def receive_tensor(
        shared_tensor: TensorShare,
        operation: Callable = Depends(get_computation_operation),
    ) -> Any:
        """Synchronous endpoint to handle tensors reception and computation."""
        result = operation(shared_tensor)

        return result
    ```

=== "Async"

    ``` python
    @router.post(
        f"{config.receive_tensor.path}",
        response_model=config.response_model,
        status_code=http_status.HTTP_200_OK,
        tags=["tensorshare"],
    )
    async def receive_tensor(
        shared_tensor: TensorShare,
        operation: Callable[[TensorShare], Awaitable[Any]] = Depends(
            get_computation_operation
        ),
    ) -> Any:
        """Asynchronous endpoint to handle tensors reception and computation."""
        result = await operation(shared_tensor)

        return result
    ```

Just like the `ping` endpoint, the endpoint path will be the one defined in the [`TensorShareServer`](../usage/tensorshare_server) configuration object.

### Customizing the computation operation

As you can see there is an `operation` parameter that is a `Callable` object.
This parameter is a dependency that will be resolved by the `get_computation_operation`
function.

It allows you to easily customize the computation operation that will be applied to the received tensor, simply by
providing a custom `Callable` object during the router creation (See [Customize the router](#customize-the-router)).

Here is the behavior of the `get_computation_operation` function:

=== "Sync"

    ``` python
    target_operation: Callable = custom_operation or default_compute_operation

    def get_computation_operation() -> Callable:
        """Dependency that returns the currently set computation function."""
        return target_operation
    ```

    By default the `target_operation` variable is set to the `default_compute_operation` function, which is defined as follows:

    ``` python
    def default_compute_operation(tensors: TensorShare) -> DefaultResponse:
        """Default action to perform when receiving a TensorShare object."""
        return DefaultResponse(message="Success!")
    ```

=== "Async"

    ``` python
    target_operation: Callable[[TensorShare], Awaitable[Any]] = (
        custom_operation or default_async_compute_operation
    )

    def get_computation_operation() -> Callable[[TensorShare], Awaitable[Any]]:
        """Dependency that returns the currently set async computation function."""
        return target_operation
    ```

    By default the `target_operation` variable is set to the `default_async_compute_operation` function, which is defined as follows:

    ``` python
    async def default_async_compute_operation(tensors: TensorShare) -> DefaultResponse:
        """Default asynchronous action to perform when receiving a TensorShare object."""
        return DefaultResponse(message="Success!")
    ```

Here is an example of a custom computation operation you could provide if you wanted to convert the received tensor
to a `torch.Tensor` and return it as a list of lists of floats:

``` python
class GoBrrBrr(BaseModel):
    """The new response_model to use for the router."""

    predictions: List[List[float]]  # List of predictions converted to list of floats

def compute_go_brr_brr(tensors: TensorShare) -> Dict[str, List[float]]:
    """New computation operation to run inference on the received tensor."""
    torch_tensors: Dict[str, torch.Tensor] = tensors.to_tensors(backend="torch")

    inference_result: List[List[float]] = []
    for _, v in torch_tensors.items():
        y_hat = ml_pred(v)
        inference_result.append(y_hat.tolist())

    return {"predictions": inference_result}
```

!!! tip
    This is a very simple example, but you can do much more complex things with this.
    You can obviously run `async` functions, or even run a whole pipeline of operations.

Now during the router creation, you can simply pass the `compute_go_brr_brr` function to the `custom_operation`
parameter and update the `response_model` parameter to match the `GoBrrBrr` model:

=== "Sync"

    ``` python
    from fastapi import FastAPI, APIRouter
    from tensorshare import create_tensorshare_router

    class GoBrrBrr(BaseModel):
        """The new response_model to use for the router."""

        predictions: List[List[float]]  # List of predictions converted to list of floats

    def compute_go_brr_brr(tensors: TensorShare) -> Dict[str, List[float]]:
        """New computation operation to run inference on the received tensor."""
        torch_tensors: Dict[str, torch.Tensor] = tensors.to_tensors(backend="torch")

        inference_result: List[List[float]] = []
        for _, v in torch_tensors.items():
            y_hat = ml_pred(v)
            inference_result.append(y_hat.tolist())

        return {"predictions": inference_result}

    app = FastAPI()  # Your FastAPI application

    server_config = {"url": "http://localhost:8000", "response_model": GoBrrBrr}

    ts_router = create_tensorshare_router(
        server_config=server_config,
        custom_operation=compute_go_brr_brr,
    )
    app.include_router(ts_router)
    ```

=== "Async"

    ``` python
    import asyncio

    from fastapi import FastAPI, APIRouter
    from tensorshare import create_async_tensorshare_router

    class GoBrrBrr(BaseModel):
        """The new response_model to use for the router."""

        predictions: List[List[float]]  # List of pridctions converted to list of floats

    async def compute_go_brr_brr(tensors: TensorShare) -> Dict[str, List[float]]:
        """New computation operation to run inference on the received tensor."""
        torch_tensors: Dict[str, torch.Tensor] = tensors.to_tensors(backend="torch")

        inference_result: List[List[float]] = []
        for _, v in torch_tensors.items():
            y_hat = await asyncio.run_in_executor(
                None, ml_pred, v
            )  # Let's say ml_pred is a synchronous inference function
            inference_result.append(y_hat.tolist())

        return {"predictions": inference_result}

    app = FastAPI()  # Your FastAPI application

    server_config = {"url": "http://localhost:8000", "response_model": GoBrrBrr}

    ts_router = create_async_tensorshare_router(
        server_config=server_config,
        custom_operation=compute_go_brr_brr,
    )
    app.include_router(ts_router)
    ```

And voilà! You now have a fully functional FastAPI application that can receive tensors from clients, run
custom operations on them and return the result.

What a breeze! 😎
