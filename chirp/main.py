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

"""Train a taxonomy classifier."""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging

from chirp import train
from chirp.data import pipeline
from ml_collections.config_flags import config_flags
import tensorflow as tf

from xmanager import xm  # pylint: disable=unused-import

_CONFIG = config_flags.DEFINE_config_file("config")
_LOGDIR = flags.DEFINE_string("logdir", None, "Work unit logging directory.")
_WORKDIR = flags.DEFINE_string("workdir", None,
                               "Work unit checkpointing directory.")
flags.mark_flags_as_required(["config", "workdir", "logdir"])


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  logging.info(_CONFIG.value)
  tf.config.experimental.set_visible_devices([], "GPU")
  train.write_config(_CONFIG.value, _WORKDIR.value)
  config = train.parse_config(_CONFIG.value)

  train_dataset, dataset_info = pipeline.get_dataset(
      "train", batch_size=config.batch_size, **config.data_config)
  valid_dataset, _ = pipeline.get_dataset(
      "test_caples", batch_size=config.batch_size, **config.data_config)
  model_bundle, train_state = train.initialize_model(
      dataset_info,
      workdir=_WORKDIR.value,
      data_config=config.data_config,
      model_config=config.model_config,
      rng_seed=config.rng_seed,
      learning_rate=config.learning_rate)
  train.train_and_evaluate(
      model_bundle,
      train_state,
      train_dataset,
      valid_dataset,
      dataset_info,
      data_config=config.data_config,
      workdir=_WORKDIR.value,
      logdir=_LOGDIR.value,
      **config.train_config)


if __name__ == "__main__":
  app.run(main)
