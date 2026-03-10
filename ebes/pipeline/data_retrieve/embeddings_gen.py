from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

# from torch import nn
import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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
        #data_df = pd.read_parquet(
        #    data_path, columns=[self.index_name, "shifts", "_seq_len", "debug_f"]
        #)
        data_df = pd.read_parquet(
            data_path, columns=[self.index_name, "shifts", "_seq_len"]
        )
        self.shifts_by_index = data_df.set_index(self.index_name)["shifts"].to_dict()
        #self.debug_f_by_index = data_df.set_index(self.index_name)["debug_f"].to_dict()
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
                    #torch.cuda.empty_cache()
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

    def get_shifts(self, old_index, offset):
        shifts = self.shifts_by_index[old_index]
        if isinstance(shifts, list):
            shifts = np.asarray(shifts)

        #debug_f = self.debug_f_by_index[old_index]
        #if isinstance(debug_f, list):
        #    debug_f = np.asarray(debug_f)

        #assert isinstance(debug_f, np.ndarray) and isinstance(
        #    shifts, np.ndarray
        #), "Provide correct types for sequential data in Dataframe."

        assert isinstance(
            shifts, np.ndarray
        ), "Provide correct types for sequential data in Dataframe."

        shifts = shifts - offset
        shifts_mask = shifts >= MIN_SHIFT_VALUE
        shifts = shifts[shifts_mask]
        #debug_f = debug_f[shifts_mask]
        #return shifts, debug_f
        return shifts

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
            offset = max(0, orig_len - old_len)

            #shifts, _ = self.get_shifts(old_index, offset)
            shifts = self.get_shifts(old_index, offset)
            shifts = np.append(shifts, int(batch.lengths[b]))
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

            emb_features_i = (
                {
                    name: torch.cat([x.unsqueeze(2) for x in lst], dim=2)
                    for name, lst in new_emb_features[i].items()
                }
                if batch.emb_features is not None
                else None
            )

            emb_mask_i = (
                {
                    name: torch.cat([x.unsqueeze(2) for x in lst], dim=2)
                    for name, lst in new_emb_mask[i].items()
                }
                if batch.emb_mask is not None
                else None
            )

            num_features_i = (
                torch.cat([x.unsqueeze(1) for x in new_num_features[i]], dim=1)
                if batch.num_features is not None and new_num_features[i]
                else None
            )

            num_mask_i = (
                torch.cat([x.unsqueeze(1) for x in new_num_mask[i]], dim=1)
                if batch.num_mask is not None and new_num_mask[i]
                else None
            )
            cat_features_i = (
                torch.cat([x.unsqueeze(1) for x in new_cat_features[i]], dim=1)
                if batch.cat_features is not None and new_cat_features[i]
                else None
            )
            cat_mask_i = (
                torch.cat([x.unsqueeze(1) for x in new_cat_mask[i]], dim=1)
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
