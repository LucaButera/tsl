from copy import deepcopy
from typing import Optional, Mapping, Union, Sequence, Dict, Tuple, Literal

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import Index
from torch import Tensor

from tsl import logger
from . import checks
from .dataset import Dataset
from .mixin import TemporalFeaturesMixin, PandasParsingMixin
from ...ops.dataframe import aggregate
from ...typing import FrameArray, OptFrameArray, IndexSlice
from ...utils.python_utils import ensure_list


class PandasDataset(Dataset, PandasParsingMixin, TemporalFeaturesMixin):
    r"""Create a tsl dataset from a :class:`pandas.DataFrame`.

    Args:
        primary (pandas.Dataframe): DataFrame containing the data related to
            the main signals. The index is considered as the temporal dimension.
            The columns are identified as:

            + *nodes*: if there is only one level (we assume the number of
              channels to be 1).

            + *(nodes, channels)*: if there are two levels (i.e., if columns is
              a :class:`~pandas.MultiIndex`). We assume nodes are at first
              level, channels at second.

        secondary (dict, optional): named mapping of DataFrame (or numpy arrays)
            with secondary data. Examples of secondary data are exogenous
            variables (in the form of multidimensional covariates) or static
            attributes (e.g., metadata). You can specify what each axis refers
            to by providing a :obj:`pattern` for each item in the mapping.
            Every item can be:

            + a :class:`~pandas.DataFrame` or :class:`~numpy.ndarray`: in this
              case the pattern is inferred from the shape (if possible).

            TODO
            (default: :obj:`None`)
        mask (pandas.Dataframe or numpy.ndarray, optional): Boolean mask
            denoting if values in data are valid (:obj:`True`) or not
            (:obj:`False`).
            (default: :obj:`None`)
        freq (str, optional): Force a sampling rate, eventually by resampling.
            (default: :obj:`None`)
        similarity_score (str): Default method to compute the similarity matrix
            with :obj:`compute_similarity`. It must be inside dataset's
            :obj:`similarity_options`.
            (default: :obj:`None`)
        temporal_aggregation (str): Default temporal aggregation method after
            resampling. This method is used during instantiation to resample the
            dataset. It must be inside dataset's
            :obj:`temporal_aggregation_options`.
            (default: :obj:`sum`)
        spatial_aggregation (str): Default spatial aggregation method for
            :obj:`aggregate`, i.e., how to aggregate multiple nodes together.
            It must be inside dataset's :obj:`spatial_aggregation_options`.
            (default: :obj:`sum`)
        default_splitting_method (str, optional): Default splitting method for
            the dataset, i.e., how to split the dataset into train/val/test.
            (default: :obj:`temporal`)
        sort_index (bool): whether to sort the dataset chronologically at
            initialization.
            (default: :obj:`True`)
        name (str, optional): Optional name of the dataset.
            (default: :obj:`class_name`)
        precision (int or str, optional): numerical precision for data: 16 (or
            "half"), 32 (or "full") or 64 (or "double").
            (default: :obj:`32`)
    """
    similarity_options = {'correntropy'}
    temporal_aggregation_options = {'sum', 'mean', 'min', 'max', 'nearest'}
    spatial_aggregation_options = {'sum', 'mean', 'min', 'max'}

    def __init__(self, primary: pd.DataFrame,
                 secondary: Optional[Mapping[str, FrameArray]] = None,
                 mask: OptFrameArray = None,
                 freq: Optional[str] = None,
                 similarity_score: Optional[str] = None,
                 temporal_aggregation: str = 'sum',
                 spatial_aggregation: str = 'sum',
                 default_splitting_method: Optional[str] = 'temporal',
                 sort_index: bool = True,
                 name: str = None,
                 precision: Union[int, str] = 32):
        super().__init__(name=name,
                         similarity_score=similarity_score,
                         temporal_aggregation=temporal_aggregation,
                         spatial_aggregation=spatial_aggregation,
                         default_splitting_method=default_splitting_method)
        # Private data buffers
        self.mask: Optional[pd.DataFrame] = None
        self._secondary = dict()

        # Set data precision before parsing objects
        self.precision = precision

        # set dataset's dataframe
        self.df: pd.DataFrame = self._parse_primary(primary, initialize=True)
        if sort_index:
            self.df.sort_index(inplace=True)

        self.set_mask(mask)

        # Store exogenous and attributes
        if secondary is not None:
            for name, value in secondary.items():
                self.add_secondary(name, **self._value_to_kwargs(value))

        # Set dataset frequency
        if freq is not None:
            self.freq = checks.to_pandas_freq(freq)
            # resample all dataframes to new frequency
            self.resample_(freq=self.freq, aggr=self.temporal_aggregation)
        else:
            try:
                freq = self.df.index.freq or self.df.index.inferred_freq
            except AttributeError:
                pass
            self.freq = None if freq is None else checks.to_pandas_freq(freq)
            self.index.freq = self.freq

    def __getattr__(self, item):
        if '_secondary' in self.__dict__ and item in self._secondary:
            return self._secondary[item]['value']
        raise AttributeError("'{}' object has no attribute '{}'"
                             .format(self.__class__.__name__, item))

    def __delattr__(self, item):
        if item == 'mask':
            self.set_mask(None)
        elif item in self._secondary:
            del self._secondary[item]
        else:
            super(PandasDataset, self).__delattr__(item)

    # Dataset properties

    @property
    def length(self) -> int:
        return self.df.values.shape[0]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def index(self) -> pd.Index:
        return self.df.index

    @property
    def nodes(self) -> pd.Index:
        return self.df.columns.unique(0)

    @property
    def channels(self) -> pd.Index:
        return self.df.columns.unique(1)

    @property
    def shape(self) -> tuple:
        return self.length, self.n_nodes, self.n_channels

    # Secondary properties

    @property
    def exogenous(self):
        return {name: attr['value'] for name, attr in self._secondary.items()
                if 't' in attr['pattern']}

    @property
    def attributes(self):
        return {name: attr['value'] for name, attr in self._secondary.items()
                if 't' not in attr['pattern']}

    # flags

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    @property
    def has_exogenous(self) -> bool:
        return len(self.exogenous) > 0

    @property
    def has_attributes(self) -> bool:
        return len(self.attributes) > 0

    # Setters #################################################################

    def set_primary(self, value: FrameArray):
        r"""Set sequence of primary channels at :obj:`self.df`."""
        self.df = self._parse_primary(value)

    def set_mask(self, mask: OptFrameArray):
        r"""Set mask of primary channels, i.e., a bool for each (node, time
        step, channel) triplet denoting if corresponding value in primary
        DataFrame is observed (1) or not (0)."""
        if mask is not None:
            mask = self._parse_primary(mask).astype('bool')
        self.mask = mask

    # Setter for secondary data

    def add_secondary(self, name: str, value: FrameArray,
                      pattern: Optional[str] = None):
        # name cannot be an attribute of self, but allow override
        invalid_names = set(dir(self))
        if name in invalid_names:
            raise ValueError(f"Cannot add object with name '{name}', "
                             f"{self.__class__.__name__} contains already an "
                             f"attribute named '{name}'.")
        value, pattern = self._parse_secondary(value, pattern)
        self._secondary[name] = dict(value=value, pattern=pattern)

    def add_exogenous(self, name: str, value: FrameArray,
                      node_level: bool = True):
        """Shortcut method to add dynamic secondary data."""
        if name.startswith('global_'):
            name = name[7:]
            node_level = False
        pattern = 't n f' if node_level else 't f'
        self.add_secondary(name, value, pattern)

    # Getters

    def get_mask(self, dtype: Union[type, str, np.dtype] = None,
                 as_numpy: bool = True):
        mask = self.mask if self.has_mask else ~self.df.isna()
        if dtype is not None:
            assert dtype in ['bool', 'uint8', bool, np.bool, np.uint8]
            mask = mask.astype(dtype)
        if as_numpy:
            mask = mask.values.reshape(self.shape)
        return mask

    def get_exogenous(self, channels: Union[Sequence, Dict] = None,
                      nodes: Sequence = None,
                      index: Sequence = None,
                      as_numpy: bool = True):
        if index is None:
            index = self.index

        if nodes is None:
            nodes = self.nodes

        # parse channels
        if channels is None:
            # defaults to all channels
            channels = self.exogenous.keys()
        elif isinstance(channels, str):
            assert channels in self.exogenous, \
                f"{channels} is not an exogenous group."
            channels = [channels]
        # expand exogenous
        if not isinstance(channels, dict):
            channels = {label: self.exogenous[label].columns.unique('channels')
                        for label in channels}
        else:
            # analyze channels dict
            for exo, chnls in channels.items():
                exo_channels = self.exogenous[exo].columns.unique('channels')
                # if value is None, default to all channels
                if chnls is None:
                    channels[exo] = exo_channels
                else:
                    chnls = ensure_list(chnls)
                    # check that all passed channels are in exo
                    wrong_channels = set(chnls).difference(exo_channels)
                    if len(wrong_channels):
                        raise KeyError(wrong_channels)

        dfs = [self.exogenous[exo].loc[index, (nodes, chnls)]
               for exo, chnls in channels.items()]

        df = pd.concat(dfs, axis=1, keys=channels.keys(),
                       names=['exogenous', 'nodes', 'channels'])
        df = df.swaplevel(i='exogenous', j='nodes', axis=1)
        # sort only nodes, keep other order as in the input variables
        df = df.loc[:, nodes]

        if as_numpy:
            return df.values.reshape((len(index), len(nodes), -1))
        return df

    # Aggregation methods

    def resample_(self, freq=None, aggr: str = None,
                  keep: Literal["first", "last", False] = 'first',
                  mask_tolerance: float = 0.):
        freq = checks.to_pandas_freq(freq) if freq is not None else self.freq
        aggr = aggr if aggr is not None else self.temporal_aggregation

        # remove duplicated steps from index
        valid_steps = ~self.index.duplicated(keep=keep)

        # aggregate mask by considering valid if average validity is higher than
        # mask_tolerance
        if self.has_mask:
            mask = self.mask[valid_steps].resample(freq)
            mask = mask.mean() >= (1. - mask_tolerance)
            self.set_mask(mask)

        self.df = self.df[valid_steps].resample(freq).apply(aggr)

        for name, attr in self._secondary.items():
            value, pattern = attr['value'], attr['pattern']
            dims = pattern.strip().split(' ')
            if dims[0] == 't':
                value = value[valid_steps].resample(freq).apply(aggr)
            for lvl, dim in enumerate(dims[1:]):
                if dim == 't':
                    value = value[valid_steps] \
                        .resample(freq, axis=1, level=lvl).apply(aggr)
            self._secondary[name]['value'] = value

        self.freq = freq

    def resample(self, freq=None, aggr: str = None,
                  keep: Literal["first", "last", False] = 'first',
                 mask_tolerance: float = 0.):
        return deepcopy(self).resample_(freq, aggr, keep, mask_tolerance)

    def aggregate_(self, node_index: Optional[Union[Index, Mapping]] = None,
                   mask_tolerance: float = 0.):

        # get aggregation function among numpy functions
        aggr_fn = getattr(np, self.spatial_aggregation)

        # node_index parsing: eventually must be an n_nodes-sized array where
        # value at position i is the cluster id of i-th node
        if node_index is None:
            # if not provided, aggregate all nodes together, with cluster id 0
            node_index = np.zeros(self.n_nodes)
        # otherwise, node_index can be a mapping {cluster_id: [nodes]}
        # the set of all nodes in mapping values must be equal to dataset nodes
        elif isinstance(node_index, Mapping):
            ids, groups = [], []
            for group_id, group in node_index.items():
                ids += [group_id] * len(group)
                groups += list(group)
            assert set(groups) == set(self.nodes)
            # reorder node_index according to nodes order in dataset
            ids, groups = np.array(ids), np.array(groups)
            _, order = np.where(self.nodes[:, None] == groups)
            node_index = ids[order]

        assert len(node_index) == self.n_nodes

        # aggregate mask (if node-wise) and threshold aggregated value
        if self.has_mask:
            mask = aggregate(self.mask, node_index, np.mean)
            mask = mask >= (1. - mask_tolerance)
            self.set_mask(mask)

        # aggregate main dataframe
        self.df = aggregate(self.df, node_index, aggr_fn)

        # aggregate all node-level exogenous
        for name, attr in self._secondary.items():
            value, pattern = attr['value'], attr['pattern']
            dims = pattern.strip().split(' ')
            if dims[0] == 'n':
                value = aggregate(value, node_index, aggr_fn, axis=0)
            for lvl, dim in enumerate(dims[1:]):
                if dim == 'n':
                    value = aggregate(value, node_index, aggr_fn,
                                      axis=1, level=lvl)
            self._secondary[name]['value'] = value

    def aggregate(self, node_index: Optional[Union[Index, Mapping]] = None,
                  mask_tolerance: float = 0.):
        return deepcopy(self).aggregate_(node_index, mask_tolerance)

    def reduce_(self, step_index=None, node_index=None):

        def index_to_mask(index, support):
            if index is None:
                return slice(None)
            elif isinstance(index, pd.Index):
                return index
            index: np.ndarray = np.asarray(index)
            if index.dtype == np.bool:
                index = support[index]
            return index

        step_index = index_to_mask(step_index, self.index)
        node_index = index_to_mask(node_index, self.nodes)
        try:
            self.df = self.df.loc[step_index, node_index]
            if self.has_mask:
                self.mask = self.mask.loc[step_index, node_index]

            for name, attr in self._secondary.items():
                value, pattern = attr['value'], attr['pattern']
                dims = pattern.strip().split(' ')
                if dims[0] == 't':
                    value = value.loc[step_index]
                elif dims[0] == 'n':
                    value = value.loc[node_index]
                for lvl, dim in enumerate(dims[1:]):
                    cols = [slice(None)] * (len(dims) - 1)
                    if dim == 't':
                        cols[lvl] = step_index
                        cols = tuple(cols) if len(cols) > 1 else cols[0]
                        value = value.loc[:, cols]
                    elif dim == 'n':
                        cols[lvl] = node_index
                        cols = tuple(cols) if len(cols) > 1 else cols[0]
                        value = value.loc[:, cols]
                self._secondary[name]['value'] = value
        except Exception as e:
            raise e
        return self

    def reduce(self, step_index=None, node_index=None):
        return deepcopy(self).reduce_(step_index, node_index)

    def cluster_(self,
                 clustering_algo,
                 clustering_kwarks,
                 sim_type='correntropy',
                 trainlen=None,
                 kn=20,
                 scale=1.):
        sim = self.get_similarity(method=sim_type, k=kn, trainlen=trainlen)
        algo = clustering_algo(**clustering_kwarks, affinity='precomputed')
        idx = algo.fit_predict(sim)
        _, counts = np.unique(idx, return_counts=True)
        logger.info(('{} ' * len(counts)).format(*counts))
        self.aggregate_(idx)
        self.df /= scale
        return self

    # Preprocessing

    def fill_missing_(self, method):
        # todo
        raise NotImplementedError()

    def detrend(self, method):
        # todo
        raise NotImplementedError()

    # Representations

    def dataframe(self) -> pd.DataFrame:
        df = self.df.reindex(index=self.index,
                             columns=self._columns_multiindex(),
                             copy=True)
        return df

    def numpy(self, return_idx=False) -> Union[ndarray, Tuple[ndarray, Index]]:
        if return_idx:
            return self.numpy(), self.index
        return self.dataframe().values.reshape(self.shape)

    def pytorch(self) -> Tensor:
        data = self.numpy()
        return torch.tensor(data)

    def copy(self) -> 'PandasDataset':
        return deepcopy(self)
