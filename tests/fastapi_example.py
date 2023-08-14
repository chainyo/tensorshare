from typing import Callable

from fastapi import Depends, FastAPI

from tensorshare.schema import DefaultResponse, TensorShare


async def compute_operation(tensors: TensorShare) -> DefaultResponse:
    """
    Default action to perform when receiving a TensorShare object with a POST request.

    Args:
        tensors (TensorShare):
            The TensorShare object received from the client.

    Returns:
        DefaultResponse:
            The response model to send back to the client with the result of the computation.
    """
    print(tensors)
    return DefaultResponse(message="Success!")


target_operation = compute_operation


def get_computation_operation() -> Callable:
    """Dependency that returns the currently set computation function."""
    return target_operation


app = FastAPI()
# app.include_router(router)


@app.post(
    "/receive_tensor",
)
async def receive_tensor(
    shared_tensor: TensorShare, operation: Callable = Depends(get_computation_operation)
):
    """Endpoint to handle tensors reception and computation."""
    r = await compute_operation(shared_tensor)
    return r


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
