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

"""Training loop for separation models."""

import functools
import os
import time
from typing import Optional

from absl import logging
from chirp.models import metrics
from chirp.models import separation_model
from clu import checkpoint
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
import flax
from flax import linen as nn
import flax.jax_utils as flax_utils
import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.experimental import jax2tf
from ml_collections import config_dict
import optax
import tensorflow as tf

EVAL_LOOP_SLEEP_S = 30


@flax.struct.dataclass
class TrainState:
  step: int
  params: flax.core.scope.VariableDict
  opt_state: optax.OptState
  model_state: flax.core.scope.FrozenVariableDict


@flax.struct.dataclass
class ModelBundle:
  model: nn.Module
  optimizer: optax.GradientTransformation
  key: jnp.ndarray
  ckpt: checkpoint.Checkpoint


def p_log_mse_loss(source: jnp.ndarray,
                   estimate: jnp.ndarray,
                   max_snr: float = 1e6,
                   **unused_kwargs):
  return lax.pmean(
      jnp.mean(metrics.log_mse_loss(source, estimate, max_snr)),
      axis_name="batch")


def p_log_snr_loss(source: jnp.ndarray,
                   estimate: jnp.ndarray,
                   max_snr: float = 1e6,
                   **unused_kwargs):
  return lax.pmean(
      jnp.mean(metrics.negative_snr_loss(source, estimate, max_snr)),
      axis_name="batch")


def p_log_sisnr_loss(source: jnp.ndarray,
                     estimate: jnp.ndarray,
                     max_snr: float = 1e6,
                     **unused_kwargs):
  return lax.pmean(
      jnp.mean(metrics.negative_snr_loss(source, estimate, max_snr=max_snr)),
      axis_name="batch")


@flax.struct.dataclass
class ValidationMetrics(clu_metrics.Collection):
  valid_loss: clu_metrics.Average.from_fun(p_log_snr_loss)
  valid_mixit_log_mse: clu_metrics.Average.from_fun(p_log_mse_loss)


@flax.struct.dataclass
class TrainingMetrics(clu_metrics.Collection):
  train_loss: clu_metrics.LastValue.from_fun(p_log_snr_loss)
  train_mixit_log_mse: clu_metrics.LastValue.from_fun(p_log_mse_loss)
  train_mixit_neg_snr: clu_metrics.LastValue.from_fun(p_log_snr_loss)


def initialize_model(input_size: int, rng_seed: int, learning_rate: float,
                     workdir: str, model_config: config_dict.ConfigDict):
  """Creates model for training, eval, or inference."""
  # Initialize random number generator
  key = random.PRNGKey(rng_seed)

  # Load model
  model_init_key, key = random.split(key)
  model = separation_model.SeparationModel(**model_config)
  variables = model.init(
      model_init_key, jnp.zeros((1, input_size)), train=False)
  model_state, params = variables.pop("params")

  # Initialize optimizer
  optimizer = optax.adam(learning_rate=learning_rate)
  opt_state = optimizer.init(params)

  # Load checkpoint
  ckpt = checkpoint.MultihostCheckpoint(workdir)
  train_state = TrainState(
      step=0, params=params, opt_state=opt_state, model_state=model_state)
  train_state = ckpt.restore_or_initialize(train_state)
  return ModelBundle(model, optimizer, key, ckpt), train_state


def train(model_bundle, train_state, train_dataset, num_train_steps: int,
          logdir: str, log_every_steps: int, checkpoint_every_steps: int,
          loss_max_snr: float) -> None:
  """Train a model."""
  train_iterator = train_dataset.as_numpy_iterator()
  initial_step = int(train_state.step)
  train_state = flax.jax_utils.replicate(train_state)
  # Logging
  writer = metric_writers.create_default_writer(logdir)
  reporter = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)

  @functools.partial(jax.pmap, axis_name="batch")
  def train_step(batch, train_state):
    """Training step for the separation model."""

    def update_step(params, model_state):
      variables = {"params": params, **model_state}
      sep_audio, model_state = model_bundle.model.apply(
          variables,
          batch["audio"],
          train=True,
          mutable=list(model_state.keys()))
      estimate, mixit_matrix = metrics.least_squares_mixit(
          reference=batch["source_audio"], estimate=sep_audio)
      train_metrics = TrainingMetrics.gather_from_model_output(
          separated=sep_audio,
          source=batch["source_audio"],
          estimate=estimate,
          mixit_matrix=mixit_matrix,
          max_snr=loss_max_snr).compute()
      return train_metrics["train_loss"], (train_metrics, model_state)

    (_, (train_metrics, model_state)), grads = jax.value_and_grad(
        update_step, has_aux=True)(train_state.params, train_state.model_state)
    grads = jax.lax.pmean(grads, axis_name="batch")
    updates, opt_state = model_bundle.optimizer.update(grads,
                                                       train_state.opt_state)
    params = optax.apply_updates(train_state.params, updates)
    train_state = TrainState(
        step=train_state.step + 1,
        params=params,
        opt_state=opt_state,
        model_state=model_state)
    return train_metrics, train_state

  for step in range(initial_step, num_train_steps + 1):
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      batch = next(train_iterator)
      train_metrics, train_state = train_step(batch, train_state)
      train_metrics = flax_utils.unreplicate(train_metrics)

      if step % log_every_steps == 0:
        writer.write_scalars(step, train_metrics)
      reporter(step)
    if step % checkpoint_every_steps == 0:
      with reporter.timed("checkpoint"):
        model_bundle.ckpt.save(flax_utils.unreplicate(train_state))
  writer.close()


def evaluate(model_bundle: ModelBundle,
             train_state: TrainState,
             valid_dataset: tf.data.Dataset,
             writer: metric_writers.MetricWriter,
             reporter: periodic_actions.ReportProgress,
             max_eval_steps: int = -1):
  """Run evaluation."""
  step = train_state.step

  @functools.partial(jax.pmap, axis_name="batch")
  def evaluate_step(valid_metrics, batch, train_state):
    variables = {"params": train_state.params, **train_state.model_state}
    sep_audio = model_bundle.model.apply(variables, batch["audio"], train=False)
    estimate, mixit_matrix = metrics.least_squares_mixit(
        reference=batch["source_audio"], estimate=sep_audio)
    return valid_metrics.merge(
        ValidationMetrics.gather_from_model_output(
            separated=sep_audio,
            source=batch["source_audio"],
            estimate=estimate,
            mixit_matrix=mixit_matrix,
            axis_name="batch"))

  with reporter.timed("eval"):
    valid_metrics = flax.jax_utils.replicate(ValidationMetrics.empty())

    for valid_step, batch in enumerate(valid_dataset.as_numpy_iterator()):
      if max_eval_steps > 0 and valid_step >= max_eval_steps:
        break
      valid_metrics = evaluate_step(valid_metrics, batch,
                                    flax_utils.replicate(train_state))

    # Log validation loss
    valid_metrics = flax_utils.unreplicate(valid_metrics)
    valid_metrics = valid_metrics.compute()

  writer.write_scalars(int(step), valid_metrics)
  writer.flush()


def evaluate_loop(model_bundle: ModelBundle,
                  train_state: TrainState,
                  valid_dataset: tf.data.Dataset,
                  workdir: str,
                  logdir: str,
                  num_train_steps: int,
                  eval_steps_per_checkpoint: int,
                  tflite_export: bool = False,
                  input_size: Optional[int] = None,
                  eval_sleep_s: int = EVAL_LOOP_SLEEP_S):
  """Run evaluation in a loop."""
  writer = metric_writers.create_default_writer(logdir)
  reporter = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)
  # Initialize last_step to -1 so we always run at least one eval.
  last_step = -1
  last_ckpt = ""

  while last_step < num_train_steps:
    ckpt = checkpoint.MultihostCheckpoint(workdir)
    if ckpt.latest_checkpoint == last_ckpt:
      time.sleep(eval_sleep_s)
      continue
    try:
      train_state = ckpt.restore_or_initialize(train_state)
    except tf.errors.NotFoundError:
      logging.warning("Checkpoint %s not found in workdir %s",
                      ckpt.latest_checkpoint, workdir)
      time.sleep(eval_sleep_s)
      continue

    evaluate(model_bundle, train_state, valid_dataset, writer, reporter,
             eval_steps_per_checkpoint)
    if tflite_export:
      export_tf_lite(model_bundle, train_state, workdir, input_size)
    last_step = int(train_state.step)
    last_ckpt = ckpt.latest_checkpoint


def export_tf_lite(model_bundle, train_state, workdir: str, input_size: int):
  """Write a TFLite flatbuffer."""
  variables = {"params": train_state.params, **train_state.model_state}

  def infer_fn(audio_batch):
    model_outputs = model_bundle.model.apply(
        variables, audio_batch, train=False)
    return model_outputs

  # TODO(tomdenton): Figure out how to support polymorphic input sizes.
  tf_predict = tf.function(
      jax2tf.convert(infer_fn, enable_xla=False),
      input_signature=[
          tf.TensorSpec(shape=[1, input_size], dtype=tf.float32, name="input")
      ],
      autograph=False)

  # Drop a saved_model while we're at it.
  tf.saved_model.save(tf_predict.get_concrete_function(),
                      os.path.join(workdir, "savedmodel"))
  with tf.io.gfile.GFile(os.path.join(workdir, "savedmodel", "ckpt.txt"),
                         "w") as f:
    f.write(f"train_state.step: {train_state.step}")

  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [tf_predict.get_concrete_function()], tf_predict)

  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
  ]
  tflite_float_model = converter.convert()

  if not tf.io.gfile.exists(workdir):
    tf.io.gfile.makedirs(workdir)
  with tf.io.gfile.GFile(os.path.join(workdir, "model.tflite"), "wb") as f:
    f.write(tflite_float_model)
  with tf.io.gfile.GFile(os.path.join(workdir, "tflite_ckpt.txt"), "w") as f:
    f.write(f"train_state.step: {train_state.step}")
