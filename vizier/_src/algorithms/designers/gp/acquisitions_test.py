# Copyright 2023 Google LLC.
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

from __future__ import annotations

"""Tests for acquisitions."""

import jax
from jax import numpy as jnp
import mock
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import gp_bandit
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.jax.optimizers import optimizers
from vizier.pyvizier import converters

from absl.testing import absltest


tfd = tfp.distributions


def _build_mock_continuous_array_specs(n):
  continuous_spec = mock.create_autospec(converters.NumpyArraySpec)
  continuous_spec.type = converters.NumpyArraySpecType.CONTINUOUS
  continuous_spec.num_dimensions = 1
  return [continuous_spec] * n


class AcquisitionsTest(absltest.TestCase):

  def test_ucb(self):
    acq = acquisitions.UCB(coefficient=2.0)
    self.assertAlmostEqual(acq(tfd.Normal(0.1, 1)), 2.1)

  def test_hvs(self):
    acq = acquisitions.HyperVolumeScalarization(coefficient=2.0)
    self.assertAlmostEqual(acq(tfd.Normal(0.1, 1)), 0.1)

  def test_ei(self):
    acq = acquisitions.EI()
    self.assertAlmostEqual(
        acq(tfd.Normal(jnp.float64(0.1), 1), labels=jnp.array([0.2])),
        0.34635347,
    )


class TrustRegionTest(absltest.TestCase):

  def test_trust_region_small(self):
    tr = acquisitions.TrustRegion(
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]),
        _build_mock_continuous_array_specs(4),
    )

    np.testing.assert_allclose(
        tr.min_linf_distance(
            np.array([
                [0., .2, .3, 0.],
                [.9, .8, .9, .9],
                [1., 1., 1., 1.],
            ]),), np.array([0.3, 0.2, 0.]))
    self.assertAlmostEqual(tr.trust_radius, 0.224, places=3)

  def test_trust_region_bigger(self):
    tr = acquisitions.TrustRegion(
        np.vstack(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
            * 10
        ),
        _build_mock_continuous_array_specs(4),
    )
    np.testing.assert_allclose(
        tr.min_linf_distance(
            np.array([
                [0., .2, .3, 0.],
                [.9, .8, .9, .9],
                [1., 1., 1., 1.],
            ]),), np.array([0.3, 0.2, 0.]))
    self.assertAlmostEqual(tr.trust_radius, 0.44, places=3)

  def test_trust_region_padded_small(self):
    # Test that padding still retrieves the same distance computations as
    # `test_trust_region_small`.
    tr = acquisitions.TrustRegion(
        np.array([
            [0.0, 0.0, 0.0, 0.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
        ]),
        _build_mock_continuous_array_specs(4),
        dimension_is_missing=np.array([False, False, False, False, True, True]),
    )

    np.testing.assert_allclose(
        tr.min_linf_distance(
            np.array([
                [0.0, 0.2, 0.3, 0.0, -100.0, 20.0],
                [0.9, 0.8, 0.9, 0.9, 23.0, 27.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, -3.0],
            ]),
        ),
        np.array([0.3, 0.2, 0.0]),
    )
    self.assertAlmostEqual(tr.trust_radius, 0.224, places=3)


class GPBanditAcquisitionBuilderTest(absltest.TestCase):

  def test_sample_on_array(self):
    ard_optimizer = optimizers.JaxoptLbfgsB(random_restarts=8, best_n=5)
    search_space = vz.SearchSpace()
    for i in range(16):
      search_space.root.add_float_param(f'x{i}', 0.0, 1.0)

    problem = vz.ProblemStatement(
        search_space=search_space,
        metric_information=vz.MetricsConfig(
            metrics=[
                vz.MetricInformation(
                    'obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE
                ),
            ]
        ),
    )
    gp_designer = gp_bandit.VizierGPBandit(problem, ard_optimizer=ard_optimizer)
    suggestions = quasi_random.QuasiRandomDesigner(
        problem.search_space
    ).suggest(11)

    trials = []
    for idx, suggestion in enumerate(suggestions):
      trial = suggestion.to_trial(idx)
      trial.complete(vz.Measurement(metrics={'obj': np.random.randn()}))
      trials.append(trial)

    gp_designer.update(vza.CompletedTrials(trials), vza.ActiveTrials())
    gp_designer._compute_state()
    xs = np.random.randn(10, 16)
    samples = gp_designer._acquisition_builder.sample_on_array(
        xs, 15, jax.random.PRNGKey(0)
    )
    self.assertEqual(samples.shape, (15, 10))
    self.assertEqual(np.sum(np.isnan(samples)), 0)


if __name__ == '__main__':
  absltest.main()
