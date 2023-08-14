from fastapi import FastAPI

from tensorshare.schema import TensorShare

app = FastAPI()


@app.post("/receive_tensor/")
def receive_tensor(shared_tensor: TensorShare):
    # async def receive_tensor(req: Request):
    """Receive a tensor from a client and print it"""
    # body = await req.json()
    # ts = TensorShare(**body)
    ts = shared_tensor
    print(type(ts), ts)

    tensors = ts.to_tensors("torch")
    print(tensors)

    return {"message": "success"}


@app.get("/ping/")
def ping():
    """Ping the server"""
    return {"message": "pong"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765)
