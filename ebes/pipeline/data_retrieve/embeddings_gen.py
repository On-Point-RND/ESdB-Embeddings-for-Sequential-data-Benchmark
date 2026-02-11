from collections.abc import Mapping
from pathlib import Path

# from torch import nn
import pandas as pd
import numpy as np

import logging
import torch
from tqdm.autonotebook import tqdm

from ...types import Batch

logger = logging.getLogger(__name__)

MIN_SHIFT_VALUE = 2


class ResultsGetter:
    def __init__(self, config, mode):
        # self.config=config
        self.mode = mode
        if mode == "train":
            data_path = Path(config["data"]["dataset"]["parquet_path"])
        elif mode == "test":
            data_path = Path(config["data"]["dataset"]["parquet_path"]).parent / "test"
        self.index_name = config["data"]["preprocessing"]["common_pipeline"][
            "index_name"
        ]
        data_df = pd.read_parquet(
            data_path, columns=[self.index_name, "shifts", "_seq_len", "debug_f"]
        )
        self.shifts_by_index = data_df.set_index(self.index_name)["shifts"].to_dict()
        self.debug_f_by_index = data_df.set_index(self.index_name)["debug_f"].to_dict()
        self.orig_len_by_index = data_df.set_index(self.index_name)[
            "_seq_len"
        ].to_dict()

    def df_get(self, loaders, trainer):
        model = trainer.model
        assert model is not None

        get_query_embeddings = getattr(model, "get_query_embeddings", None)
        df_list = []
        for loader_name, loader in loaders.items():
            if loader is None:
                raise ValueError("Incorrect loader for embeddings generation")
            logger.info(f"Embedding generation on {loader_name} started")
            records = []
            for batch_old in tqdm(loader, disable=False):
                batch = self.shift_transform(batch_old)

                batch.to(trainer.device)
                with torch.no_grad():
                    if callable(get_query_embeddings):
                        emb = get_query_embeddings(batch)
                    else:
                        emb = model(batch)

                emb_np = emb.detach().cpu().numpy()
                del emb
                torch.cuda.empty_cache()
                index_data = batch.extract_indexes_from_batch()

                for i in range(len(emb_np)):
                    record = {
                        "embedding": emb_np[i],
                        "index": index_data[i],
                    }
                    records.append(record)
            df = pd.DataFrame(records)
            df_backed = self.shift_reverse_transform(df)
            df_list.append(df_backed)
            logger.info(f"Embedding generation on {loader_name} finished")
        df_all = df_list[0]
        for i in range(1, len(df_list)):
            df_all = pd.concat([df_all, df_list[i]], ignore_index=True)
        return df_all

    def get_shifts(self, old_index, offset):
        shifts = self.shifts_by_index[old_index]
        if isinstance(shifts, list):
            shifts = np.asarray(shifts)

        debug_f = self.debug_f_by_index[old_index]
        if isinstance(debug_f, list):
            debug_f = np.asarray(debug_f)

        assert isinstance(debug_f, np.ndarray) and isinstance(
            shifts, np.ndarray
        ), "Provide correct types for sequential data in Dataframe."

        shifts = shifts - offset
        shifts_mask = shifts >= MIN_SHIFT_VALUE
        shifts = shifts[shifts_mask]
        debug_f = debug_f[shifts_mask]
        return shifts, debug_f

    def shift_transform(self, batch):
        device = batch.time.device if isinstance(batch.time, torch.Tensor) else None
        old_len, old_batch = batch.time.shape

        new_num_features = []
        new_cat_features = []
        new_num_mask = []
        new_cat_mask = []
        new_times = []
        new_lengths = []
        new_indices = []
        new_targets = []
        reps = []
        if batch.emb_features is not None:
            new_emb_features = {k: [] for k in batch.emb_features}
        if batch.emb_mask is not None:
            new_emb_mask = {k: [] for k in batch.emb_mask}
        for b in range(old_batch):
            old_index = batch.index[b]
            orig_len = int(self.orig_len_by_index[old_index])
            offset = max(0, orig_len - old_len)

            shifts, debug_f = self.get_shifts(old_index, offset)

            shifts = np.append(shifts, int(batch.lengths[b]))

            for i, s in enumerate(shifts):
                s = int(s)

                # ---- time ----
                t = batch.time[:, b]
                new_t = torch.zeros(old_len, device=device)
                new_t[:s] = t[:s]
                assert int(t[s] * 1e3) == (
                    debug_f[i]
                ), "Check shifted data for embeddings."  # TODO: Make some good debug.

                new_times.append(new_t)

                # ---- emb features ----
                if batch.emb_features is not None:
                    for name, emb in batch.emb_features.items():
                        e = emb[:, :, b]
                        new_e = torch.zeros_like(e)
                        new_e[:, :s] = e[:, :s]
                        new_emb_features[name].append(new_e)

                if batch.emb_mask is not None:
                    for name, emb in batch.emb_mask.items():
                        e = emb[:, :, b]
                        new_e = torch.zeros_like(e)
                        new_e[:, :s] = e[:, :s]
                        new_emb_mask[name].append(new_e)

                # ---- num features ----
                if batch.num_features is not None:
                    nf = batch.num_features[:, b, :]
                    new_nf = torch.zeros_like(nf)
                    new_nf[:s] = nf[:s]
                    new_num_features.append(new_nf)

                if batch.num_mask is not None:
                    nf = batch.num_mask[:, b, :]
                    new_nf = torch.zeros_like(nf)
                    new_nf[:s] = nf[:s]
                    new_num_mask.append(new_nf)

                # ---- cat features ----
                if batch.cat_features is not None:
                    cf = batch.cat_features[:, b, :]
                    new_cf = torch.zeros_like(cf)
                    new_cf[:s] = cf[:s]
                    new_cat_features.append(new_cf)

                if batch.cat_mask is not None:
                    cf = batch.cat_mask[:, b, :]
                    new_cf = torch.zeros_like(cf)
                    new_cf[:s] = cf[:s]
                    new_cat_mask.append(new_cf)

                # ---- target ----
                if batch.target is not None:
                    new_targets.append(batch.target[b])
                # ---- length ----
                new_lengths.append(s)
                # ---- index ----
                new_indices.append(f"{old_index.item()}__{s}")

        # stack
        if batch.target is not None:
            new_target = torch.stack(new_targets, dim=0)
        new_time = torch.stack(new_times, dim=1)
        new_lengths = torch.tensor(new_lengths, device=device)
        new_index = new_indices
        # print(len(new_lengths))
        assert all(
            0 <= s <= old_len for s in new_lengths
        ), f"Invalid lengths: {max(new_lengths)}, max allowed: {old_len}"

        new_emb_features = (
            {
                name: torch.cat([x.unsqueeze(2) for x in lst], dim=2)
                for name, lst in new_emb_features.items()
            }
            if batch.emb_features is not None
            else None
        )

        new_emb_mask = (
            {
                name: torch.cat([x.unsqueeze(2) for x in lst], dim=2)
                for name, lst in new_emb_mask.items()
            }
            if batch.emb_mask is not None
            else None
        )

        new_num_features = (
            torch.cat([x.unsqueeze(1) for x in new_num_features], dim=1)
            if batch.num_features is not None
            else None
        )

        new_num_mask = (
            torch.cat([x.unsqueeze(1) for x in new_num_mask], dim=1)
            if batch.num_mask is not None
            else None
        )
        new_cat_features = (
            torch.cat([x.unsqueeze(1) for x in new_cat_features], dim=1)
            if batch.cat_features is not None
            else None
        )
        new_cat_mask = (
            torch.cat([x.unsqueeze(1) for x in new_cat_mask], dim=1)
            if batch.cat_mask is not None
            else None
        )
        return Batch(
            lengths=new_lengths,
            time=new_time,
            index=new_index,
            target=None,  # если target уже задублирован по shifts — ок
            num_features=new_num_features,
            cat_features=new_cat_features,
            emb_features=new_emb_features,
            num_mask=new_num_mask,
            cat_mask=new_cat_mask,
            emb_mask=new_emb_mask,
            cat_features_names=batch.cat_features_names,
            num_features_names=batch.num_features_names,
            emb_features_names=batch.emb_features_names,
        )

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

        result["global_embedding"] = result["embeddings"].apply(lambda x: x[-1])

        result["embeddings"] = result["embeddings"].apply(lambda x: x[:-1])
        result["shifts"] = result["shifts"].apply(lambda x: x[:-1])

        return result
