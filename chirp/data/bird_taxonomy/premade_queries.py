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
from chirp.data import filter_scrub_utils as fsu

colombian_species = [
    'asctap1', 'gbwwre1', 'mattap1', 'rufant1', 'rufwre1', 'slbfin2', 'whbspi2',
    'blcjay1', 'fepowl', 'forela1', 'kebtou1', 'stbwre2', 'yeofly1', 'bubwre1',
    'recwoo1', 'rubher', 'rutjac1', 'trieup1', 'whbant1', 'whtdov', 'bobfly1',
    'compau', 'yercac1', 'chwcha1', 'ducfly', 'cocwoo1', 'linwoo1', 'yebfly1',
    'grepot1', 'laufal1', 'blcwar2', 'sthbrf8', 'trsowl', 'nobwoo1', 'blgdov1',
    'bucmot4', 'rufnig1', 'latman1', 'scrgre1', 'bugtan', 'grekis', 'ywcpar',
    'ccbfin', 'gofred1', 'sltred', 'sobtyr1', 'peptyr1', 'rumfly1', 'rawwre1',
    'blbwre1', 'crbtan1', 'canwar', 'gohman1', 'sthwoo1', 'tropar', 'whbman1',
    'whcsap1', 'whwbec1', 'blcjay2', 'whbtyr1', 'grbhaw1', 'olipic1', 'cinbec1'
]

ssw_species = [
    'amewoo', 'cangoo', 'norcar', 'amecro', 'bkcchi', 'blujay', 'brncre',
    'tuftit', 'dowwoo', 'ribgul', 'amerob', 'killde', 'rewbla', 'whbnut',
    'comgra', 'snogoo', 'mallar3', 'wooduc', 'haiwoo', 'pilwoo', 'rebwoo',
    'sonspa', 'easpho', 'norfli', 'ruckin', 'houfin', 'yebsap', 'amegfi',
    'gockin', 'bnhcow', 'rusbla', 'swaspa', 'eursta', 'belkin1', 'moudov',
    'treswa', 'bcnher', 'comyel', 'grycat', 'ovenbi1', 'purfin', 'whtspa',
    'eastow', 'grcfly', 'norwat', 'buhvir', 'naswar', 'woothr', 'veery',
    'balori', 'buwwar', 'chswar', 'grbher3', 'grnher', 'leafly', 'warvir',
    'yelwar', 'pinsis', 'robgro', 'solsan', 'boboli', 'coohaw', 'comrav',
    'reevir1', 'scatan', 'aldfly', 'daejun', 'sposan', 'tenwar', 'eawpew',
    'amered', 'easkin', 'yerwar', 'rebnut', 'brdowl', 'cedwax', 'yetvir',
    'easblu', 'houwre', 'rthhum'
]

hawaiian_species = []
chosen_ssw_species = ['amewoo', 'cangoo', 'norcar', 'killde']
held_out_ssw_species = list(set(ssw_species).difference(chosen_ssw_species))


def get_upstream_metadata_query() -> fsu.QuerySequence:
  return fsu.QuerySequence([
      fsu.Query(
          op=fsu.TransformOp.FILTER,
          kwargs={
              'mask_op': fsu.MaskOp.NOT_IN,
              'op_kwargs': {
                  'key':
                      'species_code',
                  'values':
                      colombian_species + hawaiian_species +
                      held_out_ssw_species
              }
          })
  ])


def get_upstream_data_query() -> fsu.QuerySequence:
  return fsu.QuerySequence([
      fsu.Query(
          op=fsu.TransformOp.FILTER,
          kwargs={
              'mask_op': fsu.MaskOp.NOT_IN,
              'op_kwargs': {
                  'key':
                      'species_code',
                  'values':
                      colombian_species + hawaiian_species +
                      held_out_ssw_species
              }
          }),
      fsu.Query(
          op=fsu.TransformOp.SCRUB,
          kwargs={
              'key':
                  'bg_species_codes',
              'values':
                  colombian_species + hawaiian_species + held_out_ssw_species
          }),
      fsu.QuerySequence(
          mask_query=fsu.Query(fsu.MaskOp.IN, {
              'key': 'species_code',
              'values': chosen_ssw_species
          }),
          queries=[
              fsu.Query(
                  op=fsu.TransformOp.FILTER,
                  kwargs={
                      'mask_op': fsu.MaskOp.CONTAINS_NO,
                      'op_kwargs': {
                          'key': 'bg_species_codes',
                          'values': chosen_ssw_species
                      }
                  }),
              fsu.Query(
                  fsu.TransformOp.SAMPLE_N_PER_GROUP, {
                      'group_key': 'species_code',
                      'groups': chosen_ssw_species,
                      'samples_per_group': 5,
                      'seed': 0,
                      'allow_overlap': False
                  }),
          ]),
      fsu.QuerySequence(
          mask_query=fsu.Query(fsu.MaskOp.CONTAINS_ANY, {
              'key': 'bg_species_codes',
              'values': chosen_ssw_species
          }),
          queries=[
              fsu.Query(
                  op=fsu.TransformOp.FILTER,
                  kwargs={
                      'mask_op': fsu.MaskOp.NOT_IN,
                      'op_kwargs': {
                          'key': 'species_code',
                          'values': chosen_ssw_species
                      }
                  }),
              fsu.Query(
                  fsu.TransformOp.SAMPLE_N_PER_GROUP, {
                      'group_key': 'bg_species_codes',
                      'groups': chosen_ssw_species,
                      'samples_per_group': 5,
                      'seed': 0,
                      'allow_overlap': False
                  }),
              fsu.Query(fsu.TransformOp.SCRUB_ALL_BUT, {
                  'key': 'bg_species_codes',
                  'values': chosen_ssw_species,
              })
          ])
  ])
