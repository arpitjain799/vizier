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

"""Tests for gp_bandit."""

import numpy as np
from vizier.pyvizier.converters import padding

from absl.testing import absltest
from absl.testing import parameterized


class PaddingTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          schedule=padding.PaddingSchedule(
              num_features=padding.PaddingType.NONE,
              num_dimensions=padding.PaddingType.NONE,
          ),
          num_points=17,
          num_dimensions=5,
          expected_num_points=17,
          expected_num_dimensions=5,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_features=padding.PaddingType.NONE,
              num_dimensions=padding.PaddingType.NONE,
          ),
          num_points=23,
          num_dimensions=8,
          expected_num_points=23,
          expected_num_dimensions=8,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_features=padding.PaddingType.MULTIPLES_OF_10,
              num_dimensions=padding.PaddingType.MULTIPLES_OF_10,
          ),
          num_points=17,
          num_dimensions=5,
          expected_num_points=20,
          expected_num_dimensions=10,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_features=padding.PaddingType.POWERS_OF_2,
              num_dimensions=padding.PaddingType.MULTIPLES_OF_10,
          ),
          num_points=23,
          num_dimensions=2,
          expected_num_points=32,
          expected_num_dimensions=10,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_features=padding.PaddingType.MULTIPLES_OF_10,
              num_dimensions=padding.PaddingType.MULTIPLES_OF_10,
          ),
          num_points=7,
          num_dimensions=8,
          expected_num_points=10,
          expected_num_dimensions=10,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_features=padding.PaddingType.MULTIPLES_OF_10,
              num_dimensions=padding.PaddingType.MULTIPLES_OF_10,
          ),
          num_points=7,
          num_dimensions=20,
          expected_num_points=10,
          expected_num_dimensions=20,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_features=padding.PaddingType.MULTIPLES_OF_10,
              num_dimensions=padding.PaddingType.POWERS_OF_2,
          ),
          num_points=123,
          num_dimensions=22,
          expected_num_points=130,
          expected_num_dimensions=32,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_features=padding.PaddingType.POWERS_OF_2,
              num_dimensions=padding.PaddingType.POWERS_OF_2,
          ),
          num_points=17,
          num_dimensions=5,
          expected_num_points=32,
          expected_num_dimensions=8,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_features=padding.PaddingType.POWERS_OF_2,
              num_dimensions=padding.PaddingType.POWERS_OF_2,
          ),
          num_points=23,
          num_dimensions=2,
          expected_num_points=32,
          expected_num_dimensions=2,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_features=padding.PaddingType.POWERS_OF_2,
              num_dimensions=padding.PaddingType.MULTIPLES_OF_10,
          ),
          num_points=7,
          num_dimensions=8,
          expected_num_points=8,
          expected_num_dimensions=10,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_features=padding.PaddingType.POWERS_OF_2,
              num_dimensions=padding.PaddingType.POWERS_OF_2,
          ),
          num_points=7,
          num_dimensions=17,
          expected_num_points=8,
          expected_num_dimensions=32,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_features=padding.PaddingType.POWERS_OF_2,
              num_dimensions=padding.PaddingType.POWERS_OF_2,
          ),
          num_points=123,
          num_dimensions=22,
          expected_num_points=128,
          expected_num_dimensions=32,
      ),
  )
  def test_padding(
      self,
      schedule,
      num_points,
      num_dimensions,
      expected_num_points,
      expected_num_dimensions,
  ):
    features = np.random.randn(num_points, num_dimensions)
    labels = np.random.randn(num_points)[..., np.newaxis]

    padded_features, padded_labels = padding.pad_features_and_labels(
        features, labels, schedule
    )

    self.assertLen(padded_features.is_missing, 2)
    self.assertLen(padded_labels.is_missing, 1)

    self.assertTrue(
        np.all(
            np.isclose(
                padded_features.is_missing[0], padded_labels.is_missing[0]
            )
        )
    )

    label_is_missing = padded_features.is_missing[0]
    dimension_is_missing = padded_features.is_missing[1]
    padded_features = padded_features.padded_array
    padded_labels = padded_labels.padded_array

    self.assertEqual(
        padded_features.shape, (expected_num_points, expected_num_dimensions)
    )
    self.assertEqual(dimension_is_missing.shape, (expected_num_dimensions,))
    self.assertEqual(padded_labels.shape, (expected_num_points, 1))
    self.assertEqual(label_is_missing.shape, (expected_num_points,))

    self.assertTrue(
        np.all(
            np.isclose(features, padded_features[:num_points, :num_dimensions])
        )
    )
    self.assertTrue(np.all(np.isclose(labels, padded_labels[:num_points])))
    self.assertTrue(np.all(~label_is_missing[:num_points]))
    self.assertTrue(np.all(label_is_missing[num_points:]))

    self.assertTrue(np.all(~dimension_is_missing[:num_dimensions]))
    self.assertTrue(np.all(dimension_is_missing[num_dimensions:]))


if __name__ == '__main__':
  absltest.main()
