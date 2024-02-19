# Copyright 2024 The e3x Authors.
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

import e3x
import jax.numpy as jnp
import pytest


def test_sinc() -> None:
  x = jnp.linspace(0.0, 2.0, 11)
  expected = jnp.asarray([
      [
          1.00000000e00,
          1.00000000e00,
          1.00000000e00,
          1.00000000e00,
          1.00000000e00,
          1.00000000e00,
          1.00000000e00,
          1.00000000e00,
      ],
      [
          9.71012175e-01,
          8.87063742e-01,
          7.56826639e-01,
          5.93561471e-01,
          4.13496643e-01,
          2.33872190e-01,
          7.09073991e-02,
          -6.20441660e-02,
      ],
      [
          8.87063742e-01,
          5.93561471e-01,
          2.33872190e-01,
          -6.20441660e-02,
          -2.06748337e-01,
          -1.89206615e-01,
          -6.93579093e-02,
          6.06883466e-02,
      ],
      [
          7.56826699e-01,
          2.33872294e-01,
          -1.55914947e-01,
          -1.89206675e-01,
          2.78275341e-08,
          1.26137808e-01,
          6.68206811e-02,
          -5.84680997e-02,
      ],
      [
          5.93561471e-01,
          -6.20441660e-02,
          -1.89206615e-01,
          6.06883466e-02,
          1.03374146e-01,
          -5.84681742e-02,
          -6.33616224e-02,
          5.54415472e-02,
      ],
      [
          4.13496643e-01,
          -2.06748337e-01,
          2.78275341e-08,
          1.03374146e-01,
          -8.26993659e-02,
          2.78275341e-08,
          5.90708852e-02,
          -5.16870953e-02,
      ],
      [
          2.33872294e-01,
          -1.89206675e-01,
          1.26137808e-01,
          -5.84680997e-02,
          2.78275341e-08,
          3.89786400e-02,
          -5.40590622e-02,
          4.73016798e-02,
      ],
      [
          7.09074885e-02,
          -6.93579838e-02,
          6.68206811e-02,
          -6.33616820e-02,
          5.90709560e-02,
          -5.40590622e-02,
          4.84540015e-02,
          -4.23972532e-02,
      ],
      [
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
      ],
      [
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
      ],
      [
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
      ],
  ])
  assert jnp.allclose(e3x.nn.sinc(x, num=8, limit=1.5), expected, atol=1e-5)


def test__fourier() -> None:
  x = jnp.linspace(0, 1, 11)
  expected_value = jnp.asarray([
      [1.0000000e00, 1.0000000e00, 1.0000000e00, 1.0000000e00],
      [1.0000000e00, 9.5105654e-01, 8.0901700e-01, 5.8778524e-01],
      [1.0000000e00, 8.0901700e-01, 3.0901697e-01, -3.0901703e-01],
      [1.0000000e00, 5.8778518e-01, -3.0901715e-01, -9.5105660e-01],
      [1.0000000e00, 3.0901697e-01, -8.0901706e-01, -8.0901694e-01],
      [1.0000000e00, -4.3711388e-08, -1.0000000e00, 1.1924881e-08],
      [1.0000000e00, -3.0901715e-01, -8.0901676e-01, 8.0901724e-01],
      [1.0000000e00, -5.8778518e-01, -3.0901709e-01, 9.5105660e-01],
      [1.0000000e00, -8.0901706e-01, 3.0901712e-01, 3.0901679e-01],
      [1.0000000e00, -9.5105660e-01, 8.0901724e-01, -5.8778572e-01],
      [1.0000000e00, -1.0000000e00, 1.0000000e00, -1.0000000e00],
  ])
  expected_grad = jnp.asarray(
      [
          [-0.0000000e00, -0.0000000e00, -0.0000000e00, -0.0000000e00],
          [-0.0000000e00, -9.7080559e-01, -3.6931636e00, -7.6248055e00],
          [-0.0000000e00, -1.8465818e00, -5.9756646e00, -8.9634962e00],
          [-0.0000000e00, -2.5416021e00, -5.9756641e00, -2.9124148e00],
          [-0.0000000e00, -2.9878323e00, -3.6931634e00, 5.5397468e00],
          [-0.0000000e00, -3.1415927e00, 5.4929353e-07, 9.4247780e00],
          [-0.0000000e00, -2.9878321e00, 3.6931655e00, 5.5397425e00],
          [-0.0000000e00, -2.5416019e00, 5.9756641e00, -2.9124150e00],
          [-0.0000000e00, -1.8465817e00, 5.9756641e00, -8.9634972e00],
          [-0.0000000e00, -9.7080493e-01, 3.6931617e00, -7.6248021e00],
          [-0.0000000e00, 2.7464677e-07, -1.0985871e-06, 2.2477870e-07],
      ],
  )
  value, grad = e3x.ops.evaluate_derivatives(
      lambda x: e3x.nn.functions.trigonometric._fourier(x, num=4),
      x,
      max_order=1,
  )
  assert jnp.allclose(value, expected_value, atol=1e-5)
  assert jnp.allclose(grad, expected_grad, atol=1e-5)


@pytest.mark.parametrize('num', [1, 4, 1024])
def test__fourier_has_nan_safe_derivatives(num: int) -> None:
  finfo = jnp.finfo(jnp.float32)
  x = jnp.asarray(
      [
          0.0,
          finfo.tiny,
          finfo.eps,
          0.5,
          1.0 - finfo.epsneg,
          1.0,
          1.0 + finfo.eps,
      ],
      dtype=jnp.float32,
  )
  for y in e3x.ops.evaluate_derivatives(
      lambda x: e3x.nn.functions.trigonometric._fourier(x, num=num),
      x,
      max_order=4,
  ):
    assert jnp.all(jnp.isfinite(y))


def test__fourier_raises_with_invalid_num() -> None:
  with pytest.raises(ValueError, match='num must be greater or equal to 1'):
    e3x.nn.functions.trigonometric._fourier(0, num=0)


def test_basic_fourier() -> None:
  x = jnp.linspace(0.0, 1.5, 11)
  expected = jnp.asarray([
      [
          1.0000000e00,
          1.0000000e00,
          1.0000000e00,
          1.0000000e00,
          1.0000000e00,
          1.0000000e00,
          1.0000000e00,
          1.0000000e00,
      ],
      [
          1.0000000e00,
          9.5105648e-01,
          8.0901694e-01,
          5.8778518e-01,
          3.0901685e-01,
          -1.6292068e-07,
          -3.0901715e-01,
          -5.8778560e-01,
      ],
      [
          1.0000000e00,
          8.0901694e-01,
          3.0901685e-01,
          -3.0901715e-01,
          -8.0901718e-01,
          -1.0000000e00,
          -8.0901682e-01,
          -3.0901620e-01,
      ],
      [
          1.0000000e00,
          5.8778518e-01,
          -3.0901715e-01,
          -9.5105660e-01,
          -8.0901682e-01,
          4.8876205e-07,
          8.0901724e-01,
          9.5105630e-01,
      ],
      [
          1.0000000e00,
          3.0901685e-01,
          -8.0901718e-01,
          -8.0901682e-01,
          3.0901757e-01,
          1.0000000e00,
          3.0901635e-01,
          -8.0901802e-01,
      ],
      [
          1.0000000e00,
          -4.3711388e-08,
          -1.0000000e00,
          1.1924881e-08,
          1.0000000e00,
          -3.3776624e-07,
          -1.0000000e00,
          6.6360758e-07,
      ],
      [
          1.0000000e00,
          -3.0901715e-01,
          -8.0901682e-01,
          8.0901724e-01,
          3.0901635e-01,
          -1.0000000e00,
          3.0901775e-01,
          8.0901611e-01,
      ],
      [
          1.0000000e00,
          -5.8778542e-01,
          -3.0901667e-01,
          9.5105642e-01,
          -8.0901742e-01,
          1.6172819e-06,
          8.0901664e-01,
          -9.5105702e-01,
      ],
      [
          1.0000000e00,
          -8.0901718e-01,
          3.0901757e-01,
          3.0901635e-01,
          -8.0901629e-01,
          1.0000000e00,
          -8.0901778e-01,
          3.0902019e-01,
      ],
      [
          1.0000000e00,
          -9.5105660e-01,
          8.0901724e-01,
          -5.8778578e-01,
          3.0901775e-01,
          -9.8944906e-07,
          -3.0901587e-01,
          5.8778334e-01,
      ],
      [
          1.0000000e00,
          -1.0000000e00,
          1.0000000e00,
          -1.0000000e00,
          1.0000000e00,
          -1.0000000e00,
          1.0000000e00,
          -1.0000000e00,
      ],
  ])
  assert jnp.allclose(
      e3x.nn.basic_fourier(x, num=8, limit=1.5), expected, atol=1e-5
  )


def test_reciprocal_fourier() -> None:
  x = jnp.linspace(0.0, 10.0, 11)
  expected = jnp.asarray([
      [
          1.0000000e00,
          1.0000000e00,
          1.0000000e00,
          1.0000000e00,
          1.0000000e00,
          1.0000000e00,
          1.0000000e00,
          1.0000000e00,
      ],
      [
          1.0000000e00,
          -4.3711388e-08,
          -1.0000000e00,
          1.1924881e-08,
          1.0000000e00,
          -3.3776624e-07,
          -1.0000000e00,
          6.6360758e-07,
      ],
      [
          1.0000000e00,
          -4.9999985e-01,
          -5.0000030e-01,
          1.0000000e00,
          -4.9999940e-01,
          -5.0000018e-01,
          1.0000000e00,
          -4.9999997e-01,
      ],
      [
          1.0000000e00,
          -7.0710677e-01,
          1.1924881e-08,
          7.0710677e-01,
          -1.0000000e00,
          7.0710748e-01,
          -3.5774640e-08,
          -7.0710611e-01,
      ],
      [
          1.0000000e00,
          -8.0901706e-01,
          3.0901712e-01,
          3.0901679e-01,
          -8.0901688e-01,
          1.0000000e00,
          -8.0901724e-01,
          3.0901837e-01,
      ],
      [
          1.0000000e00,
          -8.6602539e-01,
          4.9999991e-01,
          1.3907092e-07,
          -5.0000018e-01,
          8.6602527e-01,
          -1.0000000e00,
          8.6602598e-01,
      ],
      [
          1.0000000e00,
          -9.0096897e-01,
          6.2349004e-01,
          -2.2252136e-01,
          -2.2252037e-01,
          6.2348926e-01,
          -9.0096849e-01,
          1.0000000e00,
      ],
      [
          1.0000000e00,
          -9.2387962e-01,
          7.0710701e-01,
          -3.8268387e-01,
          6.6360758e-07,
          3.8268268e-01,
          -7.0710611e-01,
          9.2387909e-01,
      ],
      [
          1.0000000e00,
          -9.3969268e-01,
          7.6604468e-01,
          -5.0000024e-01,
          1.7364880e-01,
          1.7364718e-01,
          -4.9999961e-01,
          7.6604331e-01,
      ],
      [
          1.0000000e00,
          -9.5105648e-01,
          8.0901694e-01,
          -5.8778501e-01,
          3.0901682e-01,
          -3.5774640e-08,
          -3.0901769e-01,
          5.8778489e-01,
      ],
      [
          1.0000000e00,
          -9.5949298e-01,
          8.4125352e-01,
          -6.5486068e-01,
          4.1541481e-01,
          -1.4231458e-01,
          -1.4231515e-01,
          4.1541535e-01,
      ],
  ])
  assert jnp.allclose(e3x.nn.reciprocal_fourier(x, num=8), expected, atol=1e-5)


def test_exponential_fourier() -> None:
  x = jnp.linspace(0.0, 10.0, 11)
  expected = jnp.asarray([
      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      [
          1.0,
          -0.40325287,
          -0.6747742,
          0.9474622,
          -0.08935948,
          -0.87539303,
          0.7953692,
          0.23392297,
      ],
      [
          1.0,
          -0.9109694,
          0.65973043,
          -0.29101852,
          -0.12951161,
          0.5269808,
          -0.8306164,
          0.9863498,
      ],
      [
          1.0,
          -0.9877928,
          0.9514691,
          -0.89191604,
          0.8105871,
          -0.70946866,
          0.5910284,
          -0.45816016,
      ],
      [
          1.0,
          -0.998345,
          0.99338555,
          -0.985138,
          0.97362983,
          -0.958899,
          0.9409938,
          -0.91997516,
      ],
      [
          1.0,
          -0.99977595,
          0.99910396,
          -0.99798435,
          0.9964175,
          -0.99440426,
          0.9919455,
          -0.98904246,
      ],
      [
          1.0,
          -0.99996966,
          0.9998787,
          -0.99972713,
          0.99951494,
          -0.99924207,
          0.99890864,
          -0.99851465,
      ],
      [
          1.0,
          -0.9999959,
          0.9999836,
          -0.99996305,
          0.9999344,
          -0.9998974,
          0.9998523,
          -0.99979895,
      ],
      [
          1.0,
          -0.99999946,
          0.9999978,
          -0.999995,
          0.9999911,
          -0.9999861,
          0.99998003,
          -0.9999728,
      ],
      [
          1.0,
          -0.99999994,
          0.9999997,
          -0.99999934,
          0.9999988,
          -0.99999815,
          0.9999973,
          -0.9999963,
      ],
      [
          1.0,
          -1.0,
          0.99999994,
          -0.9999999,
          0.9999998,
          -0.99999976,
          0.99999964,
          -0.9999995,
      ],
  ])
  assert jnp.allclose(e3x.nn.exponential_fourier(x, num=8), expected, atol=1e-5)
