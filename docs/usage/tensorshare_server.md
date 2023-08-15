The `TensorShareServer` schema is used to define the server configuration which will receive the tensors from any client.

It will be used by the [`TensorShareClient`](../usage/tensorshare_client) for sending tensors over the network.

## Schema

The schema inherits from the [`pydantic.BaseModel`](https://docs.pydantic.dev/latest/usage/models/#basic-model-usage) class and has the following properties:

* `url`: The URL of the server. This is the URL that the client will use to connect to the server. It should be a valid URL, including the protocol (e.g. `http://localhost:5000` or `https://example.com`).

* `ping`: The path to the healthcheck endpoint. This is the path that the client will use to check if the server is available. It should be a valid string, without the leading slash (e.g. `ping` or `healthcheck`). It will be `ping` by default.

* `receive_tensor`: The path to the endpoint that will receive the tensor. This is the path that the client will use to send the tensor to the server. It should be a valid string, without the leading slash (e.g. `receive_tensor` or `tensors`). It will be `receive_tensor` by default.

* `response_model`: The response model that the server will return to the client. It should be a subclass of `BaseModel` from [Pydantic](https://docs.pydantic.dev/latest/usage/models/#basic-model-usage). It will be `DefaultResponse` by default which is a simple model returning a `message` string.

## Create a config - `from_dict`

The `from_dict` method is used to create a `TensorShareServer` config from a dictionary.

```python
from tensorshare import DefaultResponse, TensorShareServer

config = TensorShareServer.from_dict({
    "url": "http://localhost:5000",  # required
    "ping": "ping",  # optional
    "receive_tensor": "receive_tensor",  # optional
    "response_model": DefaultResponse,  # optional
})
print(config)
# url=Url('http://localhost:5000/')
# ping=Url('http://localhost:5000/ping')
# receive_tensor=Url('http://localhost:5000/receive_tensor')
# response_model=<class 'tensorshare.schema.DefaultResponse'>
```

The `url`, `ping` and `receive_tensor` properties leverages the `HttpUrl` type from [Pydantic](https://docs.pydantic.dev/latest/usage/types/urls/). This means that they will be validated and normalized.

## Use the config

Just like any pydantic model, you can use the config as a normal python object.

```python
config.url
>>> Url('http://localhost:5000/')

config.ping
>>> Url('http://localhost:5000/ping')

config.receive_tensor
>>> Url('http://localhost:5000/receive_tensor')

config.response_model
>>> <class 'tensorshare.schema.DefaultResponse'>
```

You can also get the URL as a string or some parts of it.

```python
str(config.url)
>>> 'http://localhost:5000/'

config.url.scheme
>>> 'http'

config.url.host
>>> 'localhost'

config.url.port
>>> 5000

config.url.path
>>> '/'
```
