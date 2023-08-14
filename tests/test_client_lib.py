import asyncio
import time

import aiohttp
import numpy as np
import torch

from tensorshare import TensorShare


async def test_aiohttp(data):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8765/receive_tensor",
            headers={"Content-Type": "application/json"},
            data=data,
        ) as response:
            r = await response.text()
            return r


async def main():
    aiohttp_time = []

    torcht = torch.zeros(1, 1)
    ts = TensorShare.from_dict({"embeddings": torcht})

    print(ts)
    print("\n\n")
    print(ts.model_dump())
    print("\n\n")
    print(ts.model_dump_json(), type(ts.model_dump_json()))

    n_run = 1
    for _ in range(n_run):
        start = time.time()
        await test_aiohttp(data=ts.model_dump_json())
        aiohttp_time.append(time.time() - start)

        await asyncio.sleep(0.01)

    print(f"aiohttp: {np.mean(aiohttp_time):.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
