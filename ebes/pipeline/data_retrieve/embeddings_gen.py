import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.autonotebook import tqdm

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

        self._init_time_features(config)

    def _init_time_features(self, config):
        """Set up appending explicit time features to every emitted embedding.

        The contrastive/global embedding tends to forget *when* a sequence began
        and ended. When ``append_time_features.enabled`` is set, we concatenate a
        small block of normalised time descriptors -- ``[norm_start, norm_end,
        norm_duration, log1p(n_events)]`` plus a fixed Fourier bank over the
        normalised start and end -- to each embedding row (whole-sequence and
        per-shift alike). Purely a feature-engineering step at generation time;
        the encoder and its training are untouched.
        """
        atf = config.get("append_time_features") if hasattr(config, "get") else None
        atf = atf or {}
        self.append_time = bool(atf.get("enabled", False))
        if not self.append_time:
            return

        self.n_fourier = int(atf.get("n_fourier", 8))
        time_name = config["data"]["preprocessing"]["common_pipeline"]["time_name"]
        # Normalisation constants always come from the train split, so train and
        # test embeddings share one coordinate frame (no per-split leakage/shift).
        train_path = Path(config["data"]["dataset"]["parquet_path"])
        tcol = pd.read_parquet(train_path, columns=[time_name])[time_name]
        starts = tcol.map(lambda a: a[0]).astype(float)
        ends = tcol.map(lambda a: a[-1]).astype(float)
        self.t_min = float(starts.min())
        self.t_max = float(ends.max())
        self.t_span = max(self.t_max - self.t_min, 1e-8)
        # positional-encoding style geometric frequencies over normalised [0, 1]
        self.freqs = (2.0 * np.pi) * (2.0 ** np.arange(self.n_fourier))
        logger.info(
            "append_time_features on: +%d dims (t_min=%.4g t_max=%.4g n_fourier=%d)",
            4 + 4 * self.n_fourier,
            self.t_min,
            self.t_max,
            self.n_fourier,
        )

    def _time_features(self, batch) -> np.ndarray:
        """(n_records, 4 + 4*n_fourier) block of time descriptors for a batch."""
        lens = batch.lengths.detach().cpu().numpy().astype(np.int64)
        times = batch.time.detach().cpu().numpy()  # (max_len, n_records)
        n = len(lens)
        start = times[0, :].astype(np.float64)
        end_idx = np.clip(lens - 1, 0, times.shape[0] - 1)
        end = times[end_idx, np.arange(n)].astype(np.float64)

        ns = (start - self.t_min) / self.t_span
        ne = (end - self.t_min) / self.t_span
        ndur = (end - start) / self.t_span
        logn = np.log1p(np.maximum(lens, 0).astype(np.float64))
        raw = np.stack([ns, ne, ndur, logn], axis=1)  # (n, 4)

        ang_s = np.outer(ns, self.freqs)
        ang_e = np.outer(ne, self.freqs)
        fourier = np.concatenate(
            [np.sin(ang_s), np.cos(ang_s), np.sin(ang_e), np.cos(ang_e)], axis=1
        )  # (n, 4 * n_fourier)
        return np.concatenate([raw, fourier], axis=1).astype(np.float32)

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
                        if callable(get_query_embeddings):
                            emb = get_query_embeddings(batch)
                        else:
                            emb = model(batch)
                    emb_np = emb.detach().cpu().numpy()

                    del emb
                    # torch.cuda.empty_cache()
                    if self.append_time:
                        emb_np = np.concatenate(
                            [emb_np, self._time_features(batch)], axis=1
                        )
                    index_data = batch.extract_indexes_from_batch()

                    for i in range(len(emb_np)):
                        record = {
                            "embedding": emb_np[i],
                            "index": index_data[i],
                        }
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
        # If the loader truncated the head (orig_len > full_len), drop shifts
        # that fell into the discarded prefix and translate the rest into the
        # new coordinate frame so they index into the kept tail.
        orig_len = int(self.orig_len_by_index[old_index])
        full_len_i = int(full_len)
        cutoff = max(0, orig_len - full_len_i)
        if cutoff > 0:
            shifts = shifts[shifts >= cutoff] - cutoff
        # Final safety: clamp into [0, full_len] so the downstream length
        # invariants always hold even on unusual inputs.
        shifts = np.clip(shifts, 0, full_len_i)
        return np.append(shifts, full_len_i)

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

        # Группируем
        result = tmp.groupby("base_index", as_index=False).agg(
            shifts=("shift", list), embeddings=("embedding", list)
        )

        result["global_emb"] = result["embeddings"].apply(lambda x: x[-1])
        result["shift_emb"] = result["embeddings"].apply(lambda x: x[:-1])
        result["shifts"] = result["shifts"].apply(lambda x: x[:-1])

        return result
