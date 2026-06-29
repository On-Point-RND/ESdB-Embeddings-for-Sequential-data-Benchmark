import logging
import os
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.autonotebook import tqdm

from ..representations import build_representation_extractor
from ...types import Batch

logger = logging.getLogger(__name__)

MIN_SHIFT_VALUE = 2
FASTPATH_ENV_VAR = "EBES_BERT4REC_EMB_FASTPATH"


def _fastpath_enabled() -> bool:
    value = os.getenv(FASTPATH_ENV_VAR, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _pad_seq_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot pad an empty list of tensors")
    max_len = max(t.shape[0] for t in tensors)
    out = tensors[0].new_zeros((max_len, len(tensors), *tensors[0].shape[1:]))
    for i, tensor in enumerate(tensors):
        out[: tensor.shape[0], i] = tensor
    return out


def _safe_column_part(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")


class ResultsGetter:
    def __init__(self, config, mode):
        # self.config=config
        self.mode = mode
        if mode == "train":
            data_path = Path(config["data"]["dataset"]["parquet_path"])
        elif mode == "test":
            data_path = Path(config["data"]["dataset"]["parquet_path"]).parent / "test"
        else:
            raise ValueError("Behaviour is not supported.")

        self.index_name = config["data"]["preprocessing"]["common_pipeline"][
            "index_name"
        ]
        # data_df = pd.read_parquet(
        #    data_path, columns=[self.index_name, "shifts", "_seq_len", "debug_f"]
        # )
        data_df = pd.read_parquet(
            data_path, columns=[self.index_name, "shifts", "_seq_len"]
        )
        self.shifts_by_index = data_df.set_index(self.index_name)["shifts"].to_dict()
        # self.debug_f_by_index = data_df.set_index(self.index_name)["debug_f"].to_dict()
        self.orig_len_by_index = data_df.set_index(self.index_name)[
            "_seq_len"
        ].to_dict()
        self.representation_config = config.get("representation_export", {})
        self.lidar_config = self.representation_config.get("lidar", {})
        self.representation_enabled = bool(
            self.representation_config.get("enabled", False)
        )
        self.lidar_enabled = bool(self.lidar_config.get("enabled", False))
        self.representation_extractor = (
            build_representation_extractor(self.representation_config)
            if self.representation_enabled
            else None
        )
        self._lidar_seed_offset = 0

    def df_get(self, loaders, trainer):
        model = trainer.model
        assert model is not None
        model.eval()

        get_query_embeddings = getattr(model, "get_query_embeddings", None)
        df_list = []
        for loader_name, loader in loaders.items():
            if loader is None:
                raise ValueError("Incorrect loader for embeddings generation")
            logger.info(f"Embedding generation on {loader_name} started")
            records = []
            logged_fastpath = False
            for batch_old in tqdm(loader, disable=False):
                use_fastpath = self._should_use_tail_window_fastpath(model, batch_old)
                if use_fastpath and not logged_fastpath:
                    logger.info(
                        "Embedding generation on %s uses %s tail-window fast-path",
                        loader_name,
                        model.__class__.__name__,
                    )
                    logged_fastpath = True
                if use_fastpath:
                    batches_array = self.shift_transform_tail_window(
                        batch_old, int(model.max_len)
                    )
                else:
                    batches_array = self.shift_transform(batch_old)
                # If we want to check the memory consumption
                # torch.cuda.reset_peak_memory_stats(trainer.device)
                for batch in batches_array:
                    batch.to(trainer.device)
                    with torch.no_grad():
                        if self.representation_extractor is not None:
                            representations = self._extract_representations(
                                model,
                                batch,
                            )
                        elif callable(get_query_embeddings):
                            emb = get_query_embeddings(batch)
                            representations = {"embedding": emb}
                        else:
                            emb = model(batch)
                            representations = {"embedding": emb}
                        if self.lidar_enabled:
                            representations.update(
                                self._extract_lidar_representations(
                                    model,
                                    batch,
                                    get_query_embeddings,
                                )
                            )
                    representations_np = {
                        name: value.detach().cpu().numpy()
                        for name, value in representations.items()
                    }

                    del representations
                    # torch.cuda.empty_cache()
                    index_data = batch.extract_indexes_from_batch()

                    batch_size = len(next(iter(representations_np.values())))
                    for i in range(batch_size):
                        record = {
                            "index": index_data[i],
                        }
                        for name, value in representations_np.items():
                            record[name] = value[i]
                        records.append(record)
                # If we want to check the memory consumption
                # peak_bytes = torch.cuda.max_memory_allocated(trainer.device)  # [web:1][web:8]
                # print(f"Peak allocated: {peak_bytes / 1024**2:.2f} MB")
            df = pd.DataFrame(records)
            df_backed = self.shift_reverse_transform(df)
            df_list.append(df_backed)
            logger.info(f"Embedding generation on {loader_name} finished")
        df_all = df_list[0]
        for i in range(1, len(df_list)):
            df_all = pd.concat([df_all, df_list[i]], ignore_index=True)

        return df_all

    def _extract_representations(self, model, batch: Batch, representation_config=None):
        assert self.representation_extractor is not None
        representation_config = (
            self.representation_config
            if representation_config is None
            else representation_config
        )
        representations = self.representation_extractor.extract(
            model,
            batch,
            representation_config,
        )
        if "embedding" in representations:
            return representations

        primary = representation_config.get("primary", "embedding")
        if primary in representations:
            representations["embedding"] = representations[primary]
            return representations
        if "global_emb" in representations:
            representations["embedding"] = representations["global_emb"]
            return representations
        if representation_config.get("strict", False):
            raise KeyError(
                "Representation extractor must return an 'embedding' column "
                "or configure representation_export.primary"
            )
        first_name = next(iter(representations))
        representations["embedding"] = representations[first_name]
        return representations

    def _extract_lidar_representations(
        self,
        model,
        batch: Batch,
        get_query_embeddings,
    ) -> dict[str, torch.Tensor]:
        groups = self.lidar_config.get("groups", [])
        if isinstance(groups, dict):
            groups = [groups]
        if not groups:
            groups = [self.lidar_config]

        outputs = {}
        for group in groups:
            group_name = _safe_column_part(str(group.get("name", "views")))
            n_views = int(group.get("views", self.lidar_config.get("views", 4)))
            source = group.get(
                "source",
                self.lidar_config.get(
                    "source", self.representation_config.get("primary", "embedding")
                ),
            )
            representation_config = dict(self.representation_config)
            representation_config.update(group.get("representation", {}))
            representation_config["primary"] = source
            perturbation = group.get(
                "perturbation",
                self.lidar_config.get("perturbation", "model_mask"),
            )
            params = group.get("params", self.lidar_config.get("params", {}))
            seed = int(group.get("seed", self.lidar_config.get("seed", 0)))

            for view_idx in range(n_views):
                view_batch = self._perturb_batch_for_lidar(
                    model=model,
                    batch=batch,
                    perturbation=perturbation,
                    params=params,
                    seed=seed + self._lidar_seed_offset,
                )
                self._lidar_seed_offset += 1
                if self.representation_extractor is not None:
                    representations = self._extract_representations(
                        model,
                        view_batch,
                        representation_config=representation_config,
                    )
                    if source not in representations:
                        raise KeyError(
                            f"LiDAR source {source!r} is not available in "
                            f"representations: {sorted(representations)}"
                        )
                    value = representations[source]
                elif source == "embedding" and callable(get_query_embeddings):
                    value = get_query_embeddings(view_batch)
                else:
                    raise ValueError(
                        "LiDAR export requires representation_export.enabled=true "
                        "for non-embedding sources"
                    )
                outputs[f"lidar_{group_name}_view_{view_idx}"] = value
        return outputs

    @staticmethod
    def _perturb_batch_for_lidar(
        model,
        batch: Batch,
        perturbation: str,
        params: dict,
        seed: int,
    ):
        devices = []
        if batch.lengths.device.type == "cuda":
            devices = [batch.lengths.device.index or torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if batch.lengths.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            if perturbation == "model_mask":
                mask_inputs = getattr(model, "_mask_inputs", None)
                if not callable(mask_inputs):
                    raise ValueError(
                        "LiDAR perturbation='model_mask' requires model._mask_inputs"
                    )
                perturbed_batch, _ = mask_inputs(batch)
                return perturbed_batch
            if perturbation == "window_jitter":
                return ResultsGetter._jitter_batch_window(batch, params)
        raise ValueError(f"Unknown LiDAR perturbation: {perturbation}")

    @staticmethod
    def _jitter_batch_window(batch: Batch, params: dict) -> Batch:
        max_drop_fraction = float(params.get("max_drop_fraction", 0.05))
        max_drop_events = params.get("max_drop_events")
        min_drop_events = int(params.get("min_drop_events", 0))
        min_len = int(params.get("min_len", 1))

        new_lengths = batch.lengths.clone()
        new_time = (
            torch.zeros_like(batch.time)
            if isinstance(batch.time, torch.Tensor)
            else np.zeros_like(batch.time)
        )
        new_num_features = (
            torch.zeros_like(batch.num_features)
            if batch.num_features is not None
            else None
        )
        new_cat_features = (
            torch.zeros_like(batch.cat_features)
            if batch.cat_features is not None
            else None
        )
        new_num_mask = (
            torch.zeros_like(batch.num_mask) if batch.num_mask is not None else None
        )
        new_cat_mask = (
            torch.zeros_like(batch.cat_mask) if batch.cat_mask is not None else None
        )
        new_emb_features = (
            {
                name: torch.zeros_like(value)
                for name, value in batch.emb_features.items()
            }
            if batch.emb_features is not None
            else None
        )
        if isinstance(batch.emb_mask, dict):
            new_emb_mask = {
                name: torch.zeros_like(value) for name, value in batch.emb_mask.items()
            }
        elif batch.emb_mask is not None:
            new_emb_mask = torch.zeros_like(batch.emb_mask)
        else:
            new_emb_mask = None

        for batch_idx, length in enumerate(batch.lengths):
            cur_len = int(length.item())
            max_drop = int(cur_len * max_drop_fraction)
            if max_drop_events is not None:
                max_drop = min(max_drop, int(max_drop_events))
            max_drop = min(max_drop, max(0, cur_len - min_len))
            if max_drop <= 0:
                total_drop = 0
            else:
                min_drop = min(min_drop_events, max_drop)
                total_drop = int(
                    torch.randint(
                        min_drop,
                        max_drop + 1,
                        size=(1,),
                        device=batch.lengths.device,
                    ).item()
                )

            drop_left = (
                0
                if total_drop == 0
                else int(
                    torch.randint(
                        0,
                        total_drop + 1,
                        size=(1,),
                        device=batch.lengths.device,
                    ).item()
                )
            )
            new_len = cur_len - total_drop
            src = slice(drop_left, drop_left + new_len)
            dst = slice(0, new_len)
            new_lengths[batch_idx] = new_len

            new_time[dst, batch_idx] = batch.time[src, batch_idx]
            if batch.num_features is not None:
                new_num_features[dst, batch_idx, :] = batch.num_features[
                    src, batch_idx, :
                ]
            if batch.cat_features is not None:
                new_cat_features[dst, batch_idx, :] = batch.cat_features[
                    src, batch_idx, :
                ]
            if batch.num_mask is not None:
                new_num_mask[dst, batch_idx, :] = batch.num_mask[src, batch_idx, :]
            if batch.cat_mask is not None:
                new_cat_mask[dst, batch_idx, :] = batch.cat_mask[src, batch_idx, :]
            if batch.emb_features is not None:
                for name, value in batch.emb_features.items():
                    new_emb_features[name][:, dst, batch_idx] = value[:, src, batch_idx]
            if isinstance(batch.emb_mask, dict):
                for name, value in batch.emb_mask.items():
                    new_emb_mask[name][:, dst, batch_idx] = value[:, src, batch_idx]
            elif batch.emb_mask is not None:
                new_emb_mask[dst, batch_idx, :] = batch.emb_mask[src, batch_idx, :]

        return replace(
            batch,
            lengths=new_lengths,
            time=new_time,
            num_features=new_num_features,
            cat_features=new_cat_features,
            emb_features=new_emb_features,
            num_mask=new_num_mask,
            cat_mask=new_cat_mask,
            emb_mask=new_emb_mask,
        )

    def _should_use_tail_window_fastpath(self, model, batch: Batch) -> bool:
        if not _fastpath_enabled():
            return False
        if model.__class__.__name__ not in {"Bert4Rec", "JEPA"}:
            return False
        if not hasattr(model, "max_len"):
            return False
        if batch.emb_features is not None or batch.emb_mask is not None:
            return False
        return True

    def get_shifts(self, old_index, full_len):
        shifts = self.shifts_by_index[old_index]
        if isinstance(shifts, list):
            shifts = np.asarray(shifts)

        # debug_f = self.debug_f_by_index[old_index]
        # if isinstance(debug_f, list):
        #    debug_f = np.asarray(debug_f)

        # assert isinstance(debug_f, np.ndarray) and isinstance(
        #    shifts, np.ndarray
        # ), "Provide correct types for sequential data in Dataframe."
        # debug_f = debug_f[shifts_mask]
        # return shifts, debug_f
        return np.append(shifts, int(full_len))

    def shift_transform(self, batch):
        device = batch.time.device if isinstance(batch.time, torch.Tensor) else None
        old_len, old_batch = batch.time.shape

        new_num_features = defaultdict(list)
        new_cat_features = defaultdict(list)
        new_num_mask = defaultdict(list)
        new_cat_mask = defaultdict(list)
        new_times = defaultdict(list)
        new_lengths = defaultdict(list)
        new_indices = defaultdict(list)
        if batch.emb_features is not None:
            new_emb_features = defaultdict(
                lambda: {name: [] for name in batch.emb_features}
            )
        else:
            new_emb_features = None
        if batch.emb_mask is not None:
            new_emb_mask = defaultdict(lambda: {name: [] for name in batch.emb_mask})
        else:
            new_emb_mask = None

        for b in range(old_batch):
            old_index = batch.index[b]
            orig_len = int(self.orig_len_by_index[old_index])
            shifts = self.get_shifts(old_index, batch.lengths[b])
            assert (shifts >= (orig_len - old_len)).all(), "Shifts out of seq_len"

            for i, s in enumerate(shifts):
                s = int(s)

                # ---- time ----
                t = batch.time[:, b]
                new_t = torch.zeros(old_len, device=device)
                new_t[:s] = t[:s]
                new_times[i].append(new_t)

                # ---- emb features ----
                if batch.emb_features is not None:
                    for name, emb in batch.emb_features.items():
                        e = emb[:, :, b]
                        new_e = torch.zeros_like(e)
                        new_e[:, :s] = e[:, :s]
                        new_emb_features[i][name].append(new_e)

                if batch.emb_mask is not None:
                    for name, emb in batch.emb_mask.items():
                        e = emb[:, :, b]
                        new_e = torch.zeros_like(e)
                        new_e[:, :s] = e[:, :s]
                        new_emb_mask[i][name].append(new_e)

                # ---- num features ----
                if batch.num_features is not None:
                    nf = batch.num_features[:, b, :]
                    new_nf = torch.zeros_like(nf)
                    new_nf[:s] = nf[:s]
                    new_num_features[i].append(new_nf)

                if batch.num_mask is not None:
                    nf = batch.num_mask[:, b, :]
                    new_nf = torch.zeros_like(nf)
                    new_nf[:s] = nf[:s]
                    new_num_mask[i].append(new_nf)

                # ---- cat features ----
                if batch.cat_features is not None:
                    cf = batch.cat_features[:, b, :]
                    new_cf = torch.zeros_like(cf)
                    new_cf[:s] = cf[:s]
                    new_cat_features[i].append(new_cf)

                if batch.cat_mask is not None:
                    cf = batch.cat_mask[:, b, :]
                    new_cf = torch.zeros_like(cf)
                    new_cf[:s] = cf[:s]
                    new_cat_mask[i].append(new_cf)

                # ---- length ----
                new_lengths[i].append(s)
                # ---- index ----
                new_indices[i].append(f"{old_index}__{s}")

        # stack
        batches_array = []
        for i in sorted(new_times.keys()):
            times_i = torch.stack(new_times[i], dim=1)
            lengths_i = torch.tensor(new_lengths[i], device=device)
            assert all(
                0 <= s <= old_len for s in lengths_i
            ), f"Invalid lengths: {max(lengths_i)}, max allowed: {old_len}"

            max_len = int(lengths_i.max().item())
            times_i = times_i[:max_len, :]

            emb_features_i = (
                {
                    name: torch.cat([x.unsqueeze(2) for x in lst], dim=2)[
                        :, :max_len, :
                    ]
                    for name, lst in new_emb_features[i].items()
                }
                if batch.emb_features is not None
                else None
            )

            emb_mask_i = (
                {
                    name: torch.cat([x.unsqueeze(2) for x in lst], dim=2)[
                        :, :max_len, :
                    ]
                    for name, lst in new_emb_mask[i].items()
                }
                if batch.emb_mask is not None
                else None
            )

            num_features_i = (
                torch.cat([x.unsqueeze(1) for x in new_num_features[i]], dim=1)[
                    :max_len, :, :
                ]
                if batch.num_features is not None and new_num_features[i]
                else None
            )

            num_mask_i = (
                torch.cat([x.unsqueeze(1) for x in new_num_mask[i]], dim=1)[
                    :max_len, :, :
                ]
                if batch.num_mask is not None and new_num_mask[i]
                else None
            )
            cat_features_i = (
                torch.cat([x.unsqueeze(1) for x in new_cat_features[i]], dim=1)[
                    :max_len, :, :
                ]
                if batch.cat_features is not None and new_cat_features[i]
                else None
            )
            cat_mask_i = (
                torch.cat([x.unsqueeze(1) for x in new_cat_mask[i]], dim=1)[
                    :max_len, :, :
                ]
                if batch.cat_mask is not None and new_cat_mask[i]
                else None
            )

            batches_array.append(
                Batch(
                    lengths=lengths_i,
                    time=times_i,
                    index=new_indices[i],
                    target=None,  # если target уже задублирован по shifts — ок
                    num_features=num_features_i,
                    cat_features=cat_features_i,
                    emb_features=emb_features_i,
                    num_mask=num_mask_i,
                    cat_mask=cat_mask_i,
                    emb_mask=emb_mask_i,
                    cat_features_names=batch.cat_features_names,
                    num_features_names=batch.num_features_names,
                    emb_features_names=batch.emb_features_names,
                )
            )
        return batches_array

    def shift_transform_tail_window(self, batch: Batch, window_len: int):
        if window_len < 1:
            raise ValueError("window_len must be positive")

        device = batch.time.device if isinstance(batch.time, torch.Tensor) else None
        _, old_batch = batch.time.shape

        new_num_features = defaultdict(list)
        new_cat_features = defaultdict(list)
        new_num_mask = defaultdict(list)
        new_cat_mask = defaultdict(list)
        new_times = defaultdict(list)
        new_lengths = defaultdict(list)
        new_indices = defaultdict(list)

        for b in range(old_batch):
            old_index = batch.index[b]
            full_len = int(batch.lengths[b])
            orig_len = int(self.orig_len_by_index[old_index])
            if full_len != orig_len:
                raise ValueError(
                    "Tail-window fast-path requires full untruncated "
                    f"sequences during embedding generation, got full_len={full_len} "
                    f"and orig_len={orig_len} for index={old_index!r}"
                )

            shifts = self.get_shifts(old_index, full_len)
            time_slice = batch.time[:full_len, b]
            for i, s in enumerate(shifts):
                s = int(s)
                start = max(0, s - window_len)
                length = s - start

                new_times[i].append(time_slice[start:s].clone())

                if batch.num_features is not None:
                    new_num_features[i].append(
                        batch.num_features[start:s, b, :].clone()
                    )

                if batch.num_mask is not None:
                    new_num_mask[i].append(batch.num_mask[start:s, b, :].clone())

                if batch.cat_features is not None:
                    new_cat_features[i].append(
                        batch.cat_features[start:s, b, :].clone()
                    )

                if batch.cat_mask is not None:
                    new_cat_mask[i].append(batch.cat_mask[start:s, b, :].clone())

                new_lengths[i].append(length)
                new_indices[i].append(f"{old_index}__{s}")

        batches_array = []
        for i in sorted(new_times.keys()):
            times_i = _pad_seq_tensors(new_times[i])
            lengths_i = torch.tensor(new_lengths[i], device=device)
            num_features_i = (
                _pad_seq_tensors(new_num_features[i])
                if batch.num_features is not None and new_num_features[i]
                else None
            )
            num_mask_i = (
                _pad_seq_tensors(new_num_mask[i])
                if batch.num_mask is not None and new_num_mask[i]
                else None
            )
            cat_features_i = (
                _pad_seq_tensors(new_cat_features[i])
                if batch.cat_features is not None and new_cat_features[i]
                else None
            )
            cat_mask_i = (
                _pad_seq_tensors(new_cat_mask[i])
                if batch.cat_mask is not None and new_cat_mask[i]
                else None
            )

            batches_array.append(
                Batch(
                    lengths=lengths_i,
                    time=times_i,
                    index=new_indices[i],
                    target=None,
                    num_features=num_features_i,
                    cat_features=cat_features_i,
                    emb_features=None,
                    num_mask=num_mask_i,
                    cat_mask=cat_mask_i,
                    emb_mask=None,
                    cat_features_names=batch.cat_features_names,
                    num_features_names=batch.num_features_names,
                    emb_features_names=batch.emb_features_names,
                )
            )
        return batches_array

    def shift_reverse_transform(self, df):
        # Разбиваем index на базовый индекс и сдвиг
        tmp = df.copy()
        tmp[["base_index", "shift"]] = tmp["index"].str.rsplit("__", n=1, expand=True)
        tmp["shift"] = tmp["shift"].astype(int)

        # Сортируем по shift, чтобы порядок был корректным
        tmp = tmp.sort_values(["base_index", "shift"])

        representation_cols = [
            col for col in tmp.columns if col not in {"index", "base_index", "shift"}
        ]
        aggregations = {"shifts": ("shift", list)}
        aggregations.update({col: (col, list) for col in representation_cols})
        result = tmp.groupby("base_index", as_index=False).agg(**aggregations)

        result["embeddings"] = result["embedding"]
        result["global_emb"] = result["embedding"].apply(lambda x: x[-1])
        result["shift_emb"] = result["embedding"].apply(lambda x: x[:-1])
        result = result.drop(columns=["embedding"])
        result["shifts"] = result["shifts"].apply(lambda x: x[:-1])

        for col in representation_cols:
            if col == "embedding":
                continue
            result[f"{col}_global_emb"] = result[col].apply(lambda x: x[-1])
            result[f"{col}_shift_emb"] = result[col].apply(lambda x: x[:-1])
            result = result.drop(columns=[col])

        return result
