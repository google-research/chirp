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

"""A set of premade queries to generate stable data configs."""
import json
import logging
import os

from chirp.data import filter_scrub_utils as fsu
from etils import epath

from google3.pyglib import resources

DOWNSTREAM_SPECIES_PATH = "google3/third_party/py/chirp/data/bird_taxonomy/metadata/downstream_species.txt"
SSW_STATS_PATH = "google3/third_party/py/chirp/data/bird_taxonomy/metadata/ssw_stats.json"


def get_filepath(google3_fp: str) -> epath.Path:
  """Returns the path to the resource associated with google3_fp.

  Depending on where/how the script is executed (Colab, PAR file etc.), the
  actual path to a resource may take a different form. This function transforms
  the original google3_path into the actual path to the resource.

  Args:
    google3_fp: The google3 path to the resource.

  Returns:
    The actual path to the resource.
  """
  try:
    file_path = epath.Path(resources.GetResourceFilename(google3_fp))
  except FileNotFoundError:
    # Fall back to loading from google3 at head.
    # This is helpful for Colab usage.
    head_base = "/google/src/head/depot"
    logging.warning("Could not find taxonomy_metadata.json in resources, "
                    "trying to read from google3 head.")
    file_path = epath.Path(os.path.join(head_base, google3_fp))
  return file_path


def get_upstream_metadata_query() -> fsu.QuerySequence:

  _, _, _, held_out_ssw_species = get_artificially_rare_species_constraints(
      5, 5)
  with open(get_filepath(DOWNSTREAM_SPECIES_PATH), "r") as f:
    downstream_species = list(map(lambda x: x.strip(), f.readlines()))
  return fsu.QuerySequence([
      fsu.Query(
          op=fsu.TransformOp.FILTER,
          kwargs={
              "mask_op": fsu.MaskOp.NOT_IN,
              "op_kwargs": {
                  "key": "species_code",
                  "values": downstream_species + held_out_ssw_species
              }
          })
  ])


def get_artificially_rare_species_constraints(num_foreground: int,
                                              num_background: int):
  """Obtain feasible set of artifically rare species given constraints.

  'Artifically rare' species are species for which we ony want to sample a small
  number of foreground and background recordings. Those species will be useful
  to evaluate long-tail performance of methods. Depending on the exact number
  of fg/bg recordings, some species may not contain enough samples; we call
  those `infeasible species'. We need to know infeasible species before-hand
  so that we can (i) remove them for the label space (ii) exclude their
  recordings when searching for a valid solution that satisfies the
  above-mentioned constraints.

  Args:
    num_foreground: The number of foreground recordings we want for each
      species.
    num_background: The number of background recordings we want for each
      species.

  Returns:
    target_fg: The corrected (removing unfeasible species) dictionnary of
        foreground constraints.
    target_bg: The corrected (removing unfeasible species) dictionnary of
        foreground constraints.
    feasible_species: The set of feasible species.
    unfeasible_species: The set of unfeasible species.
  """
  with open(get_filepath(SSW_STATS_PATH), "rb") as f:
    ssw_stats = json.load(f)

  # Fix the target foreground/background for SSW species
  target_fg = {k: num_foreground for k in ssw_stats}
  target_bg = {k: num_background for k in ssw_stats}
  feasible_species = [
      s for s in ssw_stats if ssw_stats[s]["fg"] >= target_fg[s] and
      ssw_stats[s]["bg"] >= target_bg[s]
  ]

  # Re-adjust the target.
  target_fg = {k: num_foreground for k in feasible_species}
  target_bg = {k: num_background for k in feasible_species}

  unfeasible_species = list(set(ssw_stats.keys()).difference(feasible_species))
  logging.info(
      "Under constraints (num_foreground=%d, num_background=%d), %d out of %d"
      "SSW species were feasible. The following species were infeasible: %s",
      num_foreground, num_background, len(feasible_species), len(ssw_stats),
      str(unfeasible_species))

  return target_fg, target_bg, feasible_species, unfeasible_species


def get_upstream_data_query() -> fsu.QuerySequence:
  """Produces the QuerySequence to generate upstream data.

  Returns:
    The QuerySequence to apply
  """
  with open(get_filepath(SSW_STATS_PATH), "rb") as f:
    ssw_stats = json.load(f)
  target_fg, target_bg, feasible_species, held_out_ssw_species = get_artificially_rare_species_constraints(
      5, 5)
  with open(get_filepath(DOWNSTREAM_SPECIES_PATH), "r") as f:
    downstream_species = list(map(lambda x: x.strip(), f.readlines()))

  return fsu.QuerySequence([
      # Filter all samples from downstream species
      fsu.Query(
          op=fsu.TransformOp.FILTER,
          kwargs={
              "mask_op": fsu.MaskOp.NOT_IN,
              "op_kwargs": {
                  "key": "species_code",
                  "values": downstream_species + held_out_ssw_species
              }
          }),
      # Scrub annotations from downstream species
      fsu.Query(
          op=fsu.TransformOp.SCRUB,
          kwargs={
              "key": "bg_species_codes",
              "values": downstream_species + held_out_ssw_species
          }),
      # Sample AR species
      fsu.QuerySequence(
          # Here we only apply the subsampling to ssw samples
          mask_query=fsu.QueryParallel([
              fsu.Query(fsu.MaskOp.CONTAINS_ANY, {
                  "key": "bg_species_codes",
                  "values": feasible_species
              }),
              fsu.Query(fsu.MaskOp.IN, {
                  "key": "species_code",
                  "values": feasible_species
              })
          ],
                                       merge_strategy=fsu.MergeStrategy.OR),
          queries=[
              fsu.Query(
                  fsu.TransformOp.SAMPLE_UNDER_CONSTRAINTS, {
                      "species_stats": ssw_stats,
                      "target_fg": target_fg,
                      "target_bg": target_bg
                  })
          ]),
      # Scrub background annotations from background AR species recordings
      fsu.QuerySequence(
          mask_query=fsu.Query(fsu.MaskOp.CONTAINS_ANY, {
              "key": "bg_species_codes",
              "values": feasible_species
          }),
          queries=[
              fsu.Query(fsu.TransformOp.SCRUB_ALL_BUT, {
                  "key": "bg_species_codes",
                  "values": feasible_species,
              })
          ])
  ])
