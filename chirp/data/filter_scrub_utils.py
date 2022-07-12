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

"""Utilities to filter/scrub data."""
import enum
import functools
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Union

from chirp.data import sampling_utils as su
import numpy as np
import pandas as pd


class MergeStrategy(enum.Enum):
  """Strategy used to merge the results of parallel queries in QueryParallel."""
  OR = 'or'
  CONCAT_NO_DUPLICATES = 'concat_no_duplicates'


class MaskOp(enum.Enum):
  """Operations used for selecting samples.

  Takes as input a dataframe and returns boolean pd.Series corresponding to the
  selected samples.
  """
  NOT_IN = 'not_in'
  CONTAINS_NO = 'contains_no'
  CONTAINS_ANY = 'contains_any'
  IN = 'in'


class TransformOp(enum.Enum):
  """Operations that transform the dataFrame.

  Take as input a dataframe, and return an updated version of this dataframe.
  """
  SCRUB = 'scrub'
  SCRUB_ALL_BUT = 'scrub_all_but'
  FILTER = 'filter'
  SAMPLE_UNDER_CONSTRAINTS = 'sample_under_constraints'


SerializableType = Union[List[Union[int, str, bytes]], MaskOp, TransformOp,
                         Dict]


class Query(NamedTuple):
  """The main interface for processing operations.

  A query is serializable.

  It contains an operation (op), along with its kwargs. Additionally,
  for 'masking query' (when the op is a MaskOp), a complement option can be
  activated to return the complement of what the original query would have
  returned. Combined with consistent PRNG seeding, this feature makes it easy to
  partition data for training and evaluation.
  """
  op: Union[MaskOp, TransformOp]
  kwargs: Dict[str, SerializableType]


class QuerySequence(NamedTuple):
  """A sequence of Queries to be applied in sequentially.

  Contains a sequence of Query to be applied sequentially on a dataframe.
  This sequence can be targeted to a subpopulation of samples through specifying
  a mask_query (i.e. a Query whose op is a MaskOp), for instance only
  scrubbing bg_labels from a specific subset of species.
  """
  queries: Sequence[Union[Query, 'QuerySequence', 'QueryParallel']]
  mask_query: Optional[Union[Query, 'QueryParallel']] = None


class QueryParallel(NamedTuple):
  """A sequence of Queries to be applied in parallel.

  Contains a sequence of Query to be applied in parallel from a given dataframe.
  Once all queries have been independently executed, we merge the resulting df
  using the merge_strategy defined.
  """
  queries: Sequence[Union[Query, QuerySequence, 'QueryParallel']]
  merge_strategy: MergeStrategy


class QueryComplement(NamedTuple):
  """Applies the complement of a query."""
  query: Union[Query, 'QuerySequence']


def apply_complement(df: pd.DataFrame,
                     query_complement: QueryComplement) -> pd.DataFrame:
  """Applies a QueryComplement.

  If the query transforms the df into a boolean Series, we just return the
  complement of the mask. For transform operations, we compare the indexes of
  samples initially present, minus samples remaining after the operation.
  This assumes that the query **does not** modify the indexing system of rows
  (with a .reset_index()) for instance.

  Args:
    df: The dataframe to apply the QueryComplement on.
    query_complement: The QueryComplement to apply.

  Returns:
    The complemented query.
  """

  updated_df = APPLY_FN[type(query_complement.query)](df,
                                                      query_complement.query)
  # If the query used a MaskOp (yields a boolean Series), we return the
  # complement of this boolean Series.
  if isinstance(query_complement.query, MaskOp):
    return ~updated_df
  # For other transformations, we use indexes to return the complement.
  else:
    if df.index.has_duplicates:
      raise RuntimeError('Complement operation relies on a unique indexing of '
                         'samples. Duplicate indexes have been detected in'
                         'the input dataframe.')
    # % TODO(mboudiaf) Find a way to ensure that the indexing system has not
    # been modified by the query. Nominally, it should never be, but we'd like
    # to make sure of this, and this can only happen through deliberate
    # operations such as .reset_index() / .set_index() etc..
    complement_indexes = set(df.index) - set(updated_df.index)
    return df.loc[complement_indexes]


def apply_query(
    df: pd.DataFrame,
    query: Query,
) -> Union[pd.DataFrame, pd.Series]:
  """Applies a query on a DataFrame.

  Args:
    df: The dataframe on which the query is applied.
    query: The query to apply.

  Returns:
    The new version of the dataFrame (or Series) after applying the query.
  """
  return OPS[query.op](df, **query.kwargs)


def apply_sequence(
    df: pd.DataFrame,
    query_sequence: QuerySequence,
) -> Union[pd.DataFrame, pd.Series]:
  """Applies a QuerySequence to a DataFrame.

  Args:
    df: The DataFrame on which to apply the query.
    query_sequence: The QuerySequence to apply to df.

  Returns:
    The updated version of the df, where all the queries in
    query_sequence.queries
    have been sequentially applied in the specified order.
  """
  if query_sequence.mask_query is not None:
    mask = APPLY_FN[type(query_sequence.mask_query)](df,
                                                     query_sequence.mask_query)
    assert mask.dtype == bool
    modifiable_df = df[mask]
    frozen_df = df[~mask]
    for query in query_sequence.queries:
      modifiable_df = APPLY_FN[type(query)](modifiable_df, query)
    return pd.concat([frozen_df, modifiable_df])
  else:
    for query in query_sequence.queries:
      df = APPLY_FN[type(query)](df, query)
    return df


def apply_parallel(
    df: pd.DataFrame,
    query_parallel: QueryParallel,
) -> Union[pd.DataFrame, pd.Series]:
  """Applies a QueryParallel to a DataFrame.

  Args:
    df: The DataFrame on which to apply the query.
    query_parallel: The QueryParallel to apply to df.

  Returns:
    The updated version of the df, where all the queries in
    query_sequence.queries
    have been sequentially applied in the specified order.
  """
  all_dfs = []
  for query in query_parallel.queries:
    all_dfs.append(APPLY_FN[type(query)](df, query))

  final_df = MERGE_FN[query_parallel.merge_strategy](all_dfs)
  return final_df


def is_in(feature_dict: Dict[str, Any], key: str,
          values: List[SerializableType]) -> bool:
  """Ensures if feature_dict[key] is in values.

  Useful for filtering.

  Args:
    feature_dict: A dictionary that represents the row (=recording) to be
      potentially filtered in a DataFrame.
    key: The field from feature_dict used for filtering.
    values: The set of values that feature_dict[key] needs to be belong to in
      order to trigger a True response.

  Returns:
    True if feature_dict[key] is in values, False otherwise.
  """
  if key not in feature_dict:
    raise ValueError(f'{key} is not a correct field. Please choose among'
                     f'{list(feature_dict.keys())}')
  expected_type = type(feature_dict[key])
  for index, val in enumerate(values):
    if not isinstance(val, expected_type):
      raise TypeError(
          'Values[{}] has type {}, while feature_dict[{}] has type {}'.format(
              index, type(val), key, expected_type))
  return feature_dict[key] in values


def contains_any(feature_dict: Dict[str, Any], key: str,
                 values: List[SerializableType]) -> bool:
  """Checks if feature_dict[key] contains any of the values.

  Args:
    feature_dict: A dictionary that represents a recording (row in the dataframe
      of audios). Note that feature_dict[key] must be a Sequence, e.g. the
      background labels.
    key: The field from feature_dict used to check.
    values: list of values such that if any element of this list is in
      feature_dict[key], the function returns True

  Returns:
    True if any value in values is in feature_dict[key] , False otherwise.
  """
  return not contains_no(feature_dict, key, values)


def contains_no(feature_dict: Dict[str, Any], key: str,
                values: List[SerializableType]) -> bool:
  """Checks that feature_dict[key] contains none of the values.

  Args:
    feature_dict: A dictionary that represents a recording (row in the dataframe
      of audios). Note that feature_dict[key] must be a Sequence, e.g. the
      background labels.
    key: The field from feature_dict used to check.
    values: The values that must not be in feature_dict[key] in order to trigger
      a True response.

  Returns:
    True if feature_dict[key] contains no element in values, False otherwise.
  """
  if key not in feature_dict:
    raise ValueError(f'{key} is not a correct field. Please choose among'
                     f'{list(feature_dict.keys())}')
  if not isinstance(feature_dict[key], Sequence):
    raise TypeError(
        f'{feature_dict[key]} must be a Sequence to check if it contains anything.'
    )
  return all([v not in feature_dict[key] for v in values])


def is_not_in(feature_dict: Dict[str, Any], key: str,
              values: List[SerializableType]) -> bool:
  return not is_in(feature_dict, key, values)


def scrub(feature_dict: Dict[str, Any],
          key: str,
          values: Sequence[SerializableType],
          all_but: bool = False) -> Dict[str, Any]:
  """Removes any occurence of any value in values from feature_dict[key].

  Args:
    feature_dict: A dictionary that represents the row (=recording) to be
      potentially scrubbed in a DataFrame.
    key: The field from feature_dict used for scrubbing.
    values: The values that will be scrubbed from feature_dict[key].
    all_but: If activated, will scrub every value, except those specified.

  Returns:
    A copy of feature_dict, where all values at key have been scrubbed.
  """

  if key not in feature_dict:
    raise ValueError(f'{key} is not a correct field.'
                     f'Please choose among {list(feature_dict.keys())}')
  if type(feature_dict[key]) not in [list, np.ndarray]:
    raise TypeError('Can only scrub values from lists/ndarrays. Current column'
                    'is of type {}'.format(type(feature_dict[key])))
  # Using this 'dirty' syntax because values and feature_dict[key] could be
  # list or ndarray -> using the 'not values' to check emptiness does not work.
  if len(values) == 0 or len(feature_dict[key]) == 0:
    return feature_dict
  data_type = type(feature_dict[key][0])
  for index, val in enumerate(values):
    if not isinstance(val, data_type):
      raise TypeError(
          'Values[{}] has type {}, while values in feature_dict[{}] have type {}'
          .format(index, type(val), key, data_type))
  # Avoid changing the feature_dict in-place.
  new_feature_dict = feature_dict.copy()
  if all_but:
    new_feature_dict[key] = [x for x in feature_dict[key] if x in values]
  else:
    new_feature_dict[key] = [x for x in feature_dict[key] if x not in values]
  return new_feature_dict


def filter_df(df: pd.DataFrame, mask_op: MaskOp,
              op_kwargs: Dict[str, SerializableType]):
  """Filters a dataframe based on the output of the mask_op.

  Args:
    df: The dataframe to be filtered.
    mask_op: The operation that generates the binary mask used for filtering.
    op_kwargs: kwargs to be passed to the mask_op.

  Returns:
    The filtered dataframe
  """
  mask_query = Query(op=mask_op, kwargs=op_kwargs)
  return df[APPLY_FN[type(mask_query)](df, mask_query)]


def or_series(series_list: List[pd.Series]) -> pd.Series:
  """Performs an OR operation on a list of boolean pd.Series.

  Args:
    series_list: List of boolean pd.Series to perform OR on.

  Returns:
    The result of s_1 or ... or s_N, for s_i in series_list.
  """
  reference_indexes = series_list[0].index
  if any([not series.index.equals(reference_indexes) for series in series_list
         ]):
    raise RuntimeError('OR operation expects consistent Series as input')
  if any([series.dtype != bool for series in series_list]):
    raise RuntimeError('OR operation expects boolean Series as input.')
  return functools.reduce(lambda s1, s2: s1.add(s2), series_list)


def concat_no_duplicates(df_list: List[pd.DataFrame]) -> pd.DataFrame:
  """Concatenates dataframes in df_list, then removes duplicates examples.

  Args:
    df_list: The list of dataframes to concatenate.

  Returns:
    The concatenated dataframe, where potential duplicated rows have been
    dropped.
  """
  reference_columns = set(df_list[0].columns)
  if any([set(df.columns) != reference_columns for df in df_list]):
    raise RuntimeError('Concatenation expects dataframes to share the exact '
                       'same set of columns.')
  concat_df = pd.concat(df_list)
  # List and np.ndarray are not hashable, therefore the method
  # .drop_duplicates() will raise an error if any of the value is of this type.
  # Instead convert to tuples for the sake of duplicate verification.
  non_duplicate_indexes = concat_df.applymap(lambda e: tuple(e) if type(
      e) in [list, np.ndarray] else e).reset_index().drop_duplicates().index
  return concat_df.iloc[non_duplicate_indexes]


APPLY_FN = {
    Query: apply_query,
    QuerySequence: apply_sequence,
    QueryComplement: apply_complement,
    QueryParallel: apply_parallel,
}

MERGE_FN = {
    MergeStrategy.OR: or_series,
    MergeStrategy.CONCAT_NO_DUPLICATES: concat_no_duplicates
}

OPS = {
    MaskOp.IN:
        lambda df, **kwargs: df.apply(
            functools.partial(is_in, **kwargs), axis=1, result_type='expand'),
    MaskOp.CONTAINS_NO:
        lambda df, **kwargs: df.apply(
            functools.partial(contains_no, **kwargs),
            axis=1,
            result_type='expand'),
    MaskOp.CONTAINS_ANY:
        lambda df, **kwargs: df.apply(
            functools.partial(contains_any, **kwargs),
            axis=1,
            result_type='expand'),
    MaskOp.NOT_IN:
        lambda df, **kwargs: df.apply(
            functools.partial(is_not_in, **kwargs),
            axis=1,
            result_type='expand'),
    TransformOp.SAMPLE_UNDER_CONSTRAINTS:
        su.sample_recordings_under_constraints,
    TransformOp.SCRUB:
        lambda df, **kwargs: df.apply(
            functools.partial(scrub, **kwargs), axis=1, result_type='expand'),
    TransformOp.SCRUB_ALL_BUT:
        lambda df, **kwargs: df.apply(
            functools.partial(functools.partial(scrub, all_but=True), **kwargs),
            axis=1,
            result_type='expand'),
    TransformOp.FILTER:
        filter_df
}
