# Copyright 2023 Thomas Chaigneau. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pydantic schemas to enable resilient and secure tensor sharing."""

from pydantic import BaseModel, ByteSize, ConfigDict


class TensorShare(BaseModel):
    """Base model for tensor sharing."""

    tensors: bytes
    size: ByteSize

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "tensors": (
                b'H\x00\x00\x00\x00\x00\x00\x00{"embeddings":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}}'
                b"       \x00\x00\x00\x00"
            ),
            "size": 84,
        },
        ser_json_bytes="base64",
    )
