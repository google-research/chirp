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

"""Configuration to run baseline model."""
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  config = config_dict.ConfigDict()
  config.batch_size = 64
  config.rng_seed = 0
  config.learning_rate = 0.01
  config.sample_rate_hz = 32000
  config.mixin_prob = 0.75
  # config.input_size is added automatically during parsing.

  train_config = config_dict.ConfigDict()
  train_config.num_train_steps = 250_000
  train_config.log_every_steps = 250
  train_config.checkpoint_every_steps = 5_000

  eval_config = config_dict.ConfigDict()
  eval_config.eval_steps_per_loop = -1
  eval_config.tflite_export = False

  data_config = config_dict.ConfigDict()
  data_config.window_size_s = 5
  data_config.min_gain = 0.15
  data_config.max_gain = 0.25
  data_config.highpass_cutoff_hz = 0.0

  model_config = config_dict.ConfigDict()
  model_config.bandwidth = 0
  model_config.band_stride = 0
  model_config.random_low_pass = False
  model_config.robust_normalization = False
  model_config.encoder_ = 'efficientnet-b1'
  model_config.taxonomy_loss_weight = 0.25

  melspec_config = config_dict.ConfigDict()
  melspec_config.melspec_depth = 160
  melspec_config.melspec_frequency = 100
  melspec_config.scaling = 'pcen'
  melspec_config.use_tf_stft = False

  config.data_config = data_config
  config.model_config = model_config
  config.model_config.melspec_config = melspec_config
  config.train_config = train_config
  config.eval_config = eval_config

  return config
