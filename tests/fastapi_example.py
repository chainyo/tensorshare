import asyncio
from typing import Dict, List

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from tensorshare import (
    TensorShare,
    create_async_tensorshare_router,
)


def ml_pred(x: torch.Tensor) -> torch.Tensor:
    """Dummy function to simulate a synchronous inference function."""
    return x.mean(dim=1)


class GoBrrBrr(BaseModel):
    """The new response_model to use for the router."""

    predictions: List[List[float]]  # List of pridctions converted to list of floats


async def compute_go_brr_brr(tensors: TensorShare) -> Dict[str, List[float]]:
    """New computation operation to run inference on the received tensor."""
    torch_tensors: Dict[str, torch.Tensor] = tensors.to_tensors(backend="torch")

    inference_result: List[List[float]] = []
    for _, v in torch_tensors.items():
        y_hat = await asyncio.run_in_executor(None, ml_pred, v)
        inference_result.append(y_hat.tolist())

    return {"predictions": inference_result}


app = FastAPI()  # Your FastAPI application

server_config = {"url": "http://localhost:8000", "response_model": GoBrrBrr}

ts_router = create_async_tensorshare_router(
    server_config=server_config,
    custom_operation=compute_go_brr_brr,
)
app.include_router(ts_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8765)
