# coding=utf-8
# Copyright 2022 The Chirp Authors.
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

"""Tests for train."""

import os
import tempfile

from chirp import config_utils
from chirp import train
from chirp.configs import baseline
from chirp.configs import config_globals
from chirp.data import pipeline
from chirp.models import efficientnet
from chirp.models import frontend
from chirp.tests import fake_dataset
from clu import checkpoint
from flax import linen as nn
from jax import numpy as jnp
from ml_collections import config_dict
import tensorflow as tf

from absl.testing import absltest


class ConstantEncoder(nn.Module):
  """A no-op encoder for quickly testing train+test loops."""

  output_dim: int = 32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:  # pylint: disable=redefined-outer-name
    return jnp.zeros([inputs.shape[0], self.output_dim])


class TrainTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.train_dir = tempfile.TemporaryDirectory("train_dir").name

    self.data_dir = tempfile.TemporaryDirectory("data_dir").name
    fake_builder = fake_dataset.FakeDataset(data_dir=self.data_dir)
    fake_builder.download_and_prepare()
    self.builder = fake_builder

  def _get_test_dataset(self, config):
    ds, dataset_info = pipeline.get_dataset(
        "train",
        dataset_directory=self.builder.data_dir,
        **config.train_data_config)
    return ds, dataset_info

  def _get_test_config(self, use_const_encoder=False) -> config_dict.ConfigDict:
    """Create configuration dictionary for training."""
    config = baseline.get_config()
    config = config_utils.parse_config(config, config_globals.get_globals())

    batch_size = config_dict.FieldReference(1)
    window_size_s = config_dict.FieldReference(1)
    mixin_prob = config_dict.FieldReference(0.)

    config.sample_rate_hz = 11_025

    config.train_data_config.batch_size = batch_size
    config.train_data_config.window_size_s = window_size_s
    config.train_data_config.mixin_prob = mixin_prob

    config.eval_data_config.batch_size = batch_size
    config.eval_data_config.window_size_s = window_size_s
    config.eval_data_config.mixin_prob = mixin_prob

    config.train_config.num_train_steps = 1
    config.train_config.log_every_steps = 1
    config.train_config.checkpoint_every_steps = 1
    config.eval_config.eval_steps_per_checkpoint = 1
    if use_const_encoder:
      config.init_config.model_config.encoder = ConstantEncoder(output_dim=32)
    else:
      config.init_config.model_config.encoder = efficientnet.EfficientNet(
          efficientnet.EfficientNetModel.B0)

    config.init_config.model_config.frontend = frontend.MelSpectrogram(
        features=32,
        stride=32_000 // 25,
        kernel_size=2_560,
        sample_rate=32_000,
        freq_range=(60, 10_000))
    return config

  def test_config_field_reference(self):
    config = self._get_test_config()
    self.assertEqual(config.train_config.num_train_steps,
                     config.eval_config.num_train_steps)

  def test_export_model(self):
    # NOTE: This test might fail when run on a machine that has a GPU but when
    # CUDA is not linked (JAX will detect the GPU so jax2tf will try to create
    # a TF graph on the GPU and fail)
    config = self._get_test_config()
    config.init_config.model_config.frontend.use_tf_stft = True

    _, dataset_info = self._get_test_dataset(config)

    model_bundle, train_state = train.initialize_model(
        dataset_info, workdir=self.train_dir, **config.init_config)

    train.export_tf_lite(model_bundle, train_state, self.train_dir,
                         config.init_config.input_size)
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(self.train_dir, "model.tflite")))

  def test_init_baseline(self):
    # Ensure that we can initialize the model with the baseline config.
    config = self._get_test_config()
    _, dataset_info = self._get_test_dataset(config)

    model_bundle, train_state = train.initialize_model(
        dataset_info, workdir=self.train_dir, **config.init_config)
    self.assertIsNotNone(model_bundle)
    self.assertIsNotNone(train_state)

  def test_train_one_step(self):
    config = self._get_test_config(use_const_encoder=True)
    ds, dataset_info = self._get_test_dataset(config)
    model_bundle, train_state = train.initialize_model(
        dataset_info, workdir=self.train_dir, **config.init_config)

    train.train(
        model_bundle=model_bundle,
        train_state=train_state,
        train_dataset=ds,
        logdir=self.train_dir,
        **config.train_config)
    ckpt = checkpoint.MultihostCheckpoint(self.train_dir)
    self.assertIsNotNone(ckpt.latest_checkpoint)

  def test_eval_one_step(self):
    config = self._get_test_config(use_const_encoder=True)
    ds, dataset_info = self._get_test_dataset(config)
    model_bundle, train_state = train.initialize_model(
        dataset_info, workdir=self.train_dir, **config.init_config)
    # Write a chekcpoint, or else the eval will hang.
    model_bundle.ckpt.save(train_state)

    config.eval_config.num_train_steps = 0
    train.evaluate_loop(
        model_bundle=model_bundle,
        train_state=train_state,
        valid_dataset=ds,
        workdir=self.train_dir,
        logdir=self.train_dir,
        eval_sleep_s=0,
        **config.eval_config)
    ckpt = checkpoint.MultihostCheckpoint(self.train_dir)
    self.assertIsNotNone(ckpt.latest_checkpoint)


if __name__ == "__main__":
  absltest.main()
