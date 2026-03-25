from collections.abc import Mapping, Sequence
from typing import Any

from torch.utils.data import DataLoader

from .accessors import InMemoryPandasDataAccessor, PandasDataAccessor
from .datasets import SeriesDataset, SizedSeriesDataset
from .loading import SequenceCollator
from . import batch_tfs


def get_accessor(
    parquet_path: str,
    split_sizes: Sequence[float],
    split_by_col: str | list[str] | None = None,
    random_split: bool = False,
    split_seed: int | None = None,
):
    return InMemoryPandasDataAccessor(
        parquet_path=parquet_path,
        split_sizes=split_sizes,
        split_by_col=split_by_col,
        random_split=random_split,
        split_seed=split_seed,
    )


def get_collator(
    time_name: str,
    cat_cardinalities: Mapping[str, int] | None = None,
    emb_dimensionalities: Mapping[str, int] | None = None,
    num_names: list[str] | None = None,
    index_name: str | None = None,
    target_name: str | list[str] | None = None,
    max_seq_len: int = 0,
    batch_transforms: Mapping[str, Any] | None = None,
    padding_type: str = "zeros",
) -> SequenceCollator:
    tfs = None
    def create_instances_from_module(
        module,
        configs: (
            list[Mapping[str, Any] | str] | Mapping[str, Mapping[str, Any] | str] | None
        ) = None,
        common_kwargs: dict = None,
    ) -> list[Any] | None:
        common_kwargs = common_kwargs or dict()
        instances = None
        if configs is not None:
            if isinstance(configs, Mapping):
                configs = configs.values()
            instances = []
            for config in configs:
                if config is None:
                    continue
                if isinstance(config, str):
                    instances.append(getattr(module, config)(**common_kwargs))
                    continue

                for class_name, params in config.items():
                    klass = getattr(module, class_name)
                    if isinstance(params, Mapping):
                        instances.append(klass(**(params | common_kwargs)))
                    else:
                        raise TypeError("Class config has to be mapping")
                    break  # Only process first key-value pair in dict
        return instances

    tfs = create_instances_from_module(
        module=batch_tfs,
        configs=batch_transforms
    )

    return SequenceCollator(
        time_name=time_name,
        cat_cardinalities=cat_cardinalities,
        emb_dimensionalities=emb_dimensionalities,
        num_names=num_names,
        index_name=index_name,
        target_name=target_name,
        max_seq_len=max_seq_len,
        batch_transforms=tfs,
        padding_type=padding_type,
    )


def get_loader(
    accessor: PandasDataAccessor,
    collators: Mapping[str, SequenceCollator],
    split_idx: int,
    preprocessing: str,
    batch_size: int,
    query: str | None = None,
    drop_incomplete: bool = False,
    shuffle: bool = False,
    loop: bool = False,
    random_seed: int | None = None,
    num_workers: int = 0,
    labeled: bool = True,
) -> DataLoader:
    # TODO add logging of completence
    data = accessor.get_split(split_idx)
    if isinstance(data, Sequence):
        raise NotImplementedError
    if labeled:
        labeled_query = f"`{collators[preprocessing].target_name}`.notna()"
        query = (labeled_query + " and " + query) if query else labeled_query
    ds = SizedSeriesDataset(
        data,
        batch_size=batch_size,
        query=query,
        drop_incomplete=drop_incomplete,
        shuffle=shuffle,
        loop=loop,
        random_seed=random_seed,
    )
    return DataLoader(
        dataset=ds,
        batch_size=None,
        collate_fn=collators[preprocessing],
        num_workers=num_workers,
        persistent_workers=num_workers > 0 and loop,
    )


def build_loaders(
    dataset: Mapping[str, Any],
    loaders: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
) -> Mapping[str, DataLoader]:
    acc = get_accessor(**dataset)
    collators = {k: get_collator(**v) for k, v in preprocessing.items()}
    return {k: get_loader(acc, collators, **v) for k, v in loaders.items()}
