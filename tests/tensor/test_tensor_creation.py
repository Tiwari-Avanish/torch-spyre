# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


@instantiate_parametrized_tests
class TestSpyreTensorCreation(TestCase):
    def setUp(self):
        torch.manual_seed(0xAFFE)

    def test_initializes(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "spyre")

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("shape", [(4, 8), (16, 32), (1024, 256), (2, 3, 4)])
    def test_zeros_like_basic(self, dtype, shape):
        """Test zeros_like creates tensor with correct shape, dtype, and all zeros"""
        x = torch.randn(shape, device="spyre", dtype=dtype)
        z = torch.zeros_like(x)

        self.assertEqual(z.shape, x.shape)
        self.assertEqual(z.dtype, x.dtype)
        self.assertEqual(z.device, x.device)
        self.assertTrue(torch.all(z.cpu() == 0))

    @parametrize("dtype", [torch.float16, torch.float32])
    def test_zeros_like_dtype_override(self, dtype):
        """Test zeros_like with dtype override"""
        x = torch.randn(4, 8, device="spyre", dtype=torch.float16)
        z = torch.zeros_like(x, dtype=dtype)

        self.assertEqual(z.dtype, dtype)
        self.assertEqual(z.shape, x.shape)
        self.assertTrue(torch.all(z.cpu() == 0))

    def test_zeros_like_compiled(self):
        """Test zeros_like in compiled context"""

        @torch.compile
        def use_zeros_like(x):
            mask = torch.zeros_like(x)
            return x + mask

        x = torch.randn(4, 8, device="spyre", dtype=torch.float16)
        result = use_zeros_like(x)

        self.assertTrue(torch.allclose(result.cpu(), x.cpu(), rtol=1e-3, atol=1e-3))

    @parametrize("dtype", [torch.float16, torch.float32])
    def test_zeros_like_empty_tensor(self, dtype):
        """Test zeros_like with empty tensor"""
        x = torch.empty(0, device="spyre", dtype=dtype)
        z = torch.zeros_like(x)

        self.assertEqual(z.shape, torch.Size([0]))
        self.assertEqual(z.dtype, dtype)
        self.assertEqual(z.device, x.device)


if __name__ == "__main__":
    run_tests()
