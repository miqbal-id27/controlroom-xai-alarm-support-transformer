"""
Utilities for SHAP-based analysis of Transformer HDFS anomaly detection.

Main use:
- parse prediction database fields,
- build sequence-level SHAP samples,
- run SHAP for selected sequences,
- run simple top-SHAP vs random-position ablation.
"""

import ast
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_list_cell(x):
    """Convert list-like dataframe cells into a list of integers."""
    if x is None:
        return []

    if isinstance(x, np.ndarray):
        values = x.reshape(-1).tolist()
        return [int(v) for v in values if pd.notna(v)]

    if isinstance(x, (list, tuple)):
        return [int(v) for v in x if pd.notna(v)]

    if isinstance(x, float) and np.isnan(x):
        return []

    if isinstance(x, (int, np.integer, float)):
        return [int(x)]

    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return []

        try:
            y = ast.literal_eval(s)
            return parse_list_cell(y)
        except Exception:
            return [int(v) for v in re.findall(r"-?\d+", s)]

    return []


def to_bool(x):
    """Convert common boolean-like values to bool."""
    if isinstance(x, bool):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def load_hdfs_windows(file_path, window_size=10, pad_id=0):
    """Load HDFS sequences into fixed-length source and target windows."""
    src_windows, tgt_windows = [], []

    with open(file_path, "r") as f:
        for line in f:
            seq = [int(v) for v in line.strip().split()]
            if not seq:
                continue

            src = seq[:window_size]
            tgt = seq[window_size: 2 * window_size]

            src = src + [pad_id] * (window_size - len(src))
            tgt = tgt + [pad_id] * (window_size - len(tgt))

            src_windows.append(src)
            tgt_windows.append(tgt)

    return np.asarray(src_windows, dtype=np.int64), np.asarray(tgt_windows, dtype=np.int64)


def infer_event_vocabulary_from_train(train_path, pad_id=0):
    """Infer PAD, BOS, vocabulary size, and seen EventIDs from training data."""
    max_event_id = 0
    unique_ids = set()

    with open(train_path, "r") as f:
        for line in f:
            for n in map(int, line.strip().split()):
                if n != pad_id:
                    unique_ids.add(n)
                    max_event_id = max(max_event_id, n)

    bos_id = max_event_id + 1

    return {
        "pad_id": pad_id,
        "max_event_id": max_event_id,
        "bos_id": bos_id,
        "vocab_size": bos_id + 1,
        "unique_event_ids": sorted(unique_ids),
    }


def build_sequence_level_semantic_df(pred_db, pad_id=0, bos_id=None):
    """Aggregate prediction-row database into one row per sequence."""
    df = pred_db.copy()

    list_cols = ["input_seq", "target_seq", "pred_events", "pred_probs"]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_cell)

    bool_cols = ["target_event_is_unseen", "target_in_topk_pred", "results"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(to_bool)

    group_cols = [c for c in ["dataset", "seq_id"] if c in df.columns]
    if not group_cols:
        df["_seq_row_id"] = np.arange(len(df))
        group_cols = ["_seq_row_id"]

    agg = {}
    first_cols = [
        "sequence", "seq_len", "input_seq", "target_seq",
        "sequence_predicted_status", "sequence_results_matrix",
    ]
    for col in first_cols:
        if col in df.columns:
            agg[col] = "first"

    optional_aggs = {
        "target_event_is_unseen": "any",
        "target_in_topk_pred": "mean",
        "target_in_pred_pos": "mean",
        "target_in_topk_pred_prob": "mean",
        "results": "mean",
    }
    for col, func in optional_aggs.items():
        if col in df.columns:
            agg[col] = func

    if "results_matrix" in df.columns:
        agg["results_matrix"] = lambda s: "|".join(sorted(set(map(str, s.dropna()))))

    seq_df = df.groupby(group_cols, dropna=False).agg(agg).reset_index()

    n_steps = (
        df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_token_rows")
    )
    seq_df = seq_df.merge(n_steps, on=group_cols, how="left")

    if "sequence_results_matrix" in seq_df.columns:
        seq_df["stratum"] = seq_df["sequence_results_matrix"].astype(str)
    elif {"dataset", "sequence_predicted_status"}.issubset(seq_df.columns):
        seq_df["stratum"] = (
            seq_df["dataset"].astype(str)
            + "_"
            + seq_df["sequence_predicted_status"].astype(str)
        )
    elif "dataset" in seq_df.columns:
        seq_df["stratum"] = seq_df["dataset"].astype(str)
    else:
        seq_df["stratum"] = "all"

    return seq_df


def make_stratified_shap_sample(
    seq_df,
    n_per_stratum=100,
    seed=42,
    priority_strata=("FP", "FN"),
    max_priority_per_stratum=None,
):
    """Sample sequence-level cases by stratum for SHAP analysis."""
    rng = np.random.default_rng(seed)
    sampled_parts = []
    priority_set = set(priority_strata)

    for stratum, part in seq_df.groupby("stratum", dropna=False):
        if str(stratum) in priority_set and max_priority_per_stratum is not None:
            n = min(max_priority_per_stratum, len(part))
        else:
            n = min(n_per_stratum, len(part))

        sampled_parts.append(
            part.sample(n=n, random_state=int(rng.integers(0, 1_000_000)))
        )

    sample = pd.concat(sampled_parts, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sample["shap_sample_id"] = np.arange(len(sample))

    return sample


def build_src(src_batch, window_size, bos_id, pad_id, device):
    """Build source tensor and source mask."""
    src_batch = np.asarray(src_batch, dtype=np.int64)

    if src_batch.ndim == 1:
        src_batch = src_batch[None, :]

    src_full = np.zeros((src_batch.shape[0], window_size + 1), dtype=np.int64)
    src_full[:, 0] = bos_id
    src_full[:, 1:] = src_batch[:, :window_size]

    src = torch.tensor(src_full, dtype=torch.long, device=device)
    src_mask = (src != pad_id).unsqueeze(-2)

    return src, src_mask


def build_decoder_input(tnsf, fixed_tgt, step_idx, batch_size, bos_id, pad_id, device):
    """Build decoder input up to the selected prediction step."""
    fixed_tgt = np.asarray(fixed_tgt, dtype=np.int64)

    ys = np.zeros((batch_size, step_idx + 1), dtype=np.int64)
    ys[:, 0] = bos_id

    if step_idx > 0:
        ys[:, 1:] = fixed_tgt[:step_idx]

    ys = torch.tensor(ys, dtype=torch.long, device=device)
    ys_mask = tnsf.subsequent_mask(ys.size(1)).to(device)

    return ys, ys_mask


def predict_one_step(
    model,
    tnsf,
    src_seq,
    tgt_seq,
    step_idx,
    top_k,
    window_size,
    bos_id,
    pad_id,
    device,
):
    """Predict one target step and return top-k decision details."""
    src_seq = np.asarray(parse_list_cell(src_seq), dtype=np.int64)[:window_size]
    tgt_seq = np.asarray(parse_list_cell(tgt_seq), dtype=np.int64)[:window_size]

    true_event = int(tgt_seq[step_idx])
    if true_event == pad_id:
        return None

    src, src_mask = build_src(src_seq, window_size, bos_id, pad_id, device)
    ys, ys_mask = build_decoder_input(
        tnsf, tgt_seq, step_idx, 1, bos_id, pad_id, device
    )

    with torch.inference_mode():
        memory = model.encode(src, src_mask)
        out = model.decode(memory, src_mask, ys, ys_mask)
        log_probs = model.generator(out[:, -1, :])

    top_values, top_ids = torch.topk(log_probs, k=top_k, dim=1)
    top_ids_np = top_ids[0].detach().cpu().numpy()
    top_values_np = top_values[0].detach().cpu().numpy()

    kth_log_prob = float(top_values_np[-1])
    true_log_prob = float(log_probs[0, true_event].item())

    return {
        "step_idx": int(step_idx),
        "true_event_id": true_event,
        "predicted_event_id": int(top_ids_np[0]),
        "true_log_prob": true_log_prob,
        "topk_boundary_log_prob": kth_log_prob,
        "topk_margin_log_prob": true_log_prob - kth_log_prob,
        "hit_top_1": int(true_event == int(top_ids_np[0])),
        "hit_top_k": int(true_event in set(top_ids_np.tolist())),
        "top_k_event_ids": [int(x) for x in top_ids_np.tolist()],
        "top_k_log_probs": [float(x) for x in top_values_np.tolist()],
    }


def inspect_token_predictions(
    model,
    tnsf,
    src_seq,
    tgt_seq,
    top_k,
    window_size,
    bos_id,
    pad_id,
    device,
):
    """Inspect all valid target positions in one sequence."""
    rows = []
    tgt_seq = parse_list_cell(tgt_seq)[:window_size]

    for step_idx in range(window_size):
        if step_idx >= len(tgt_seq) or int(tgt_seq[step_idx]) == pad_id:
            break

        row = predict_one_step(
            model=model,
            tnsf=tnsf,
            src_seq=src_seq,
            tgt_seq=tgt_seq,
            step_idx=step_idx,
            top_k=top_k,
            window_size=window_size,
            bos_id=bos_id,
            pad_id=pad_id,
            device=device,
        )
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


def make_next_token_scorer(
    model,
    tnsf,
    fixed_tgt,
    step_idx,
    top_k,
    window_size,
    bos_id,
    pad_id,
    device,
    score_type="true_log_prob",
):
    """Create the scoring function used by SHAP."""
    fixed_tgt = np.asarray(parse_list_cell(fixed_tgt), dtype=np.int64)[:window_size]

    if fixed_tgt.size == 0:
        raise ValueError("fixed_tgt is empty after parsing.")
    if step_idx >= fixed_tgt.size:
        raise ValueError(f"step_idx={step_idx} exceeds target length={fixed_tgt.size}.")

    true_event = int(fixed_tgt[step_idx])

    def score_fn(src_batch):
        src_batch = np.asarray(src_batch, dtype=np.int64)
        if src_batch.ndim == 1:
            src_batch = src_batch[None, :]

        src, src_mask = build_src(src_batch, window_size, bos_id, pad_id, device)
        ys, ys_mask = build_decoder_input(
            tnsf,
            fixed_tgt,
            step_idx,
            src_batch.shape[0],
            bos_id,
            pad_id,
            device,
        )

        with torch.inference_mode():
            memory = model.encode(src, src_mask)
            out = model.decode(memory, src_mask, ys, ys_mask)
            log_probs = model.generator(out[:, -1, :])

        true_lp = log_probs[:, true_event]

        if score_type == "true_log_prob":
            score = true_lp
        elif score_type == "topk_margin_log_prob":
            kth_lp = torch.topk(log_probs, k=top_k, dim=1).values[:, -1]
            score = true_lp - kth_lp
        else:
            raise ValueError(f"Unknown score_type: {score_type}")

        return score.detach().cpu().numpy()

    return score_fn


def select_background(background_src, background_size=50, seed=42):
    """Select a reproducible subset of background source windows."""
    background_src = np.asarray(background_src, dtype=np.int64)

    if len(background_src) <= background_size:
        return background_src

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(background_src), size=background_size, replace=False)

    return background_src[idx]


def explain_one_step_with_shap(
    model,
    tnsf,
    src_seq,
    tgt_seq,
    step_idx,
    background_src,
    top_k,
    window_size,
    bos_id,
    pad_id,
    device,
    background_size=50,
    max_evals=220,
    seed=42,
    score_type="true_log_prob",
):
    """Run SHAP for one prediction step."""
    import shap

    src_seq = np.asarray(parse_list_cell(src_seq), dtype=np.int64)[:window_size]
    tgt_seq = np.asarray(parse_list_cell(tgt_seq), dtype=np.int64)[:window_size]

    if tgt_seq.size == 0:
        raise ValueError("target sequence is empty after parsing.")
    if step_idx >= tgt_seq.size:
        raise ValueError(f"step_idx={step_idx} exceeds target length={tgt_seq.size}.")

    feature_names = [f"src_pos_{i + 1}" for i in range(window_size)]
    background = select_background(background_src, background_size, seed)

    scorer = make_next_token_scorer(
        model=model,
        tnsf=tnsf,
        fixed_tgt=tgt_seq,
        step_idx=step_idx,
        top_k=top_k,
        window_size=window_size,
        bos_id=bos_id,
        pad_id=pad_id,
        device=device,
        score_type=score_type,
    )

    masker = shap.maskers.Independent(background)
    explainer = shap.Explainer(
        scorer,
        masker,
        feature_names=feature_names,
        algorithm="permutation",
    )

    shap_values = explainer(src_seq[None, :], max_evals=max_evals)

    values = np.asarray(shap_values[0].values, dtype=float).reshape(-1)
    base_values = np.asarray(shap_values[0].base_values, dtype=float).reshape(-1)
    base_value = float(base_values[0]) if len(base_values) else np.nan

    abs_values = np.abs(values)
    total_abs = float(abs_values.sum())
    ranks = pd.Series(-abs_values).rank(method="first").astype(int).values

    result_df = pd.DataFrame({
        "step_idx": int(step_idx),
        "src_position": np.arange(1, window_size + 1),
        "src_event_id": src_seq,
        "true_next_event_id": int(tgt_seq[step_idx]),
        "score_type": score_type,
        "shap_value": values,
        "abs_shap_value": abs_values,
        "shap_rank": ranks,
        "shap_share": abs_values / total_abs if total_abs > 0 else np.nan,
        "base_value": base_value,
        "background_size": int(len(background)),
        "background_seed": int(seed),
        "max_evals": int(max_evals),
    })

    return result_df.sort_values("shap_rank").reset_index(drop=True), shap_values


def build_token_shap_database_for_sequence(
    model,
    tnsf,
    src_seq,
    tgt_seq,
    background_src,
    top_k,
    window_size,
    bos_id,
    pad_id,
    device,
    background_size=50,
    max_evals=220,
    seed=42,
    score_type="true_log_prob",
):
    """Run SHAP for all valid prediction steps in one sequence."""
    src_seq = parse_list_cell(src_seq)
    tgt_seq = parse_list_cell(tgt_seq)

    if not src_seq or not tgt_seq:
        return pd.DataFrame()

    prediction_df = inspect_token_predictions(
        model=model,
        tnsf=tnsf,
        src_seq=src_seq,
        tgt_seq=tgt_seq,
        top_k=top_k,
        window_size=window_size,
        bos_id=bos_id,
        pad_id=pad_id,
        device=device,
    )

    if prediction_df.empty:
        return pd.DataFrame()

    all_rows = []

    for step_idx in prediction_df["step_idx"].tolist():
        shap_df, _ = explain_one_step_with_shap(
            model=model,
            tnsf=tnsf,
            src_seq=src_seq,
            tgt_seq=tgt_seq,
            step_idx=int(step_idx),
            background_src=background_src,
            top_k=top_k,
            window_size=window_size,
            bos_id=bos_id,
            pad_id=pad_id,
            device=device,
            background_size=background_size,
            max_evals=max_evals,
            seed=seed,
            score_type=score_type,
        )

        pred_row = prediction_df[prediction_df["step_idx"] == step_idx].iloc[0].to_dict()

        pred_cols = [
            "predicted_event_id",
            "true_log_prob",
            "topk_boundary_log_prob",
            "topk_margin_log_prob",
            "hit_top_1",
            "hit_top_k",
            "top_k_event_ids",
            "top_k_log_probs",
        ]
        for col in pred_cols:
            value = pred_row[col]
            shap_df[col] = json.dumps(value) if isinstance(value, list) else value

        all_rows.append(shap_df)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def run_shap_for_sample(
    sample_df,
    model,
    tnsf,
    background_src,
    output_path,
    top_k,
    window_size,
    bos_id,
    pad_id,
    device,
    background_size=50,
    max_evals=220,
    seed=42,
    score_type="true_log_prob",
    max_sequences=None,
    save_every=10,
):
    """Run SHAP for a sequence sample and save checkpoint files."""
    output_path = Path(output_path)
    dfx = sample_df.copy()

    if max_sequences is not None:
        dfx = dfx.head(max_sequences).copy()

    rows_buffer = []
    part_paths = []
    start_time = time.time()

    for local_i, (_, row) in enumerate(dfx.iterrows(), start=1):
        src_seq = parse_list_cell(row["input_seq"])
        tgt_seq = parse_list_cell(row["target_seq"])

        if not src_seq or not tgt_seq:
            continue

        seq_seed = seed + int(row.get("shap_sample_id", local_i))

        one_db = build_token_shap_database_for_sequence(
            model=model,
            tnsf=tnsf,
            src_seq=src_seq,
            tgt_seq=tgt_seq,
            background_src=background_src,
            top_k=top_k,
            window_size=window_size,
            bos_id=bos_id,
            pad_id=pad_id,
            device=device,
            background_size=background_size,
            max_evals=max_evals,
            seed=seq_seed,
            score_type=score_type,
        )

        if one_db.empty:
            continue

        meta_cols = [
            "shap_sample_id", "dataset", "seq_id", "stratum",
            "sequence_predicted_status", "sequence_results_matrix",
            "target_event_is_unseen", "target_in_topk_pred",
            "target_in_pred_pos", "target_in_topk_pred_prob",
            "n_token_rows",
        ]
        for col in meta_cols:
            if col in row.index:
                one_db[col] = row[col]

        one_db["input_seq"] = json.dumps(src_seq)
        one_db["target_seq"] = json.dumps(tgt_seq)

        rows_buffer.append(one_db)

        if local_i % save_every == 0:
            part = pd.concat(rows_buffer, ignore_index=True)
            part_path = output_path.with_name(
                f"{output_path.stem}_part_{len(part_paths) + 1:03d}{output_path.suffix}"
            )
            save_dataframe(part, part_path)
            part_paths.append(part_path)
            rows_buffer = []
            print(f"Saved checkpoint: {part_path} | sequences processed: {local_i}")

    if rows_buffer:
        part = pd.concat(rows_buffer, ignore_index=True)
        part_path = output_path.with_name(
            f"{output_path.stem}_part_{len(part_paths) + 1:03d}{output_path.suffix}"
        )
        save_dataframe(part, part_path)
        part_paths.append(part_path)
        print(f"Saved final checkpoint: {part_path}")

    if not part_paths:
        return pd.DataFrame()

    final = pd.concat([load_dataframe(p) for p in part_paths], ignore_index=True)
    save_dataframe(final, output_path)

    elapsed = time.time() - start_time
    print(f"Final SHAP DB saved: {output_path}")
    print(f"Rows: {len(final):,} | elapsed: {elapsed:.1f}s")

    return final


def save_dataframe(df, path):
    """Save dataframe as CSV or Parquet based on file suffix."""
    path = Path(path)

    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_dataframe(path):
    """Load dataframe as CSV or Parquet based on file suffix."""
    path = Path(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def perturb_src_positions(
    src_seq,
    positions_1based,
    mode,
    rng,
    pad_id=0,
    bos_id=None,
    seen_event_ids=None,
    background_src=None,
):
    """Perturb selected source positions."""
    src_new = parse_list_cell(src_seq)

    for p1 in map(int, positions_1based):
        i = p1 - 1
        if i < 0 or i >= len(src_new):
            continue

        old = src_new[i]

        if mode == "pad":
            src_new[i] = int(pad_id)

        elif mode == "random_event":
            if not seen_event_ids:
                raise ValueError("seen_event_ids is required for random_event mode.")
            candidates = [
                int(x) for x in seen_event_ids
                if int(x) not in {pad_id, bos_id, old}
            ]
            if candidates:
                src_new[i] = int(rng.choice(candidates))

        elif mode == "background_position":
            if background_src is None:
                raise ValueError("background_src is required for background_position mode.")
            row = background_src[int(rng.integers(0, len(background_src)))]
            src_new[i] = int(row[i])

        else:
            raise ValueError(f"Unknown perturbation mode: {mode}")

    return src_new


def run_top_shap_vs_random_ablation(
    shap_db,
    sequence_df,
    model,
    tnsf,
    top_m,
    top_k,
    window_size,
    bos_id,
    pad_id,
    device,
    seen_event_ids,
    mode="random_event",
    background_src=None,
    seed=42,
    max_cases=None,
):
    """Compare perturbing top-SHAP positions against random positions."""
    rng = np.random.default_rng(seed)
    cases = []

    key_cols = [c for c in ["dataset", "seq_id"] if c in shap_db.columns and c in sequence_df.columns]
    if not key_cols:
        key_cols = ["shap_sample_id"]

    seq_lookup = sequence_df.copy()
    for col in ["input_seq", "target_seq"]:
        seq_lookup[col] = seq_lookup[col].apply(parse_list_cell)

    if "shap_sample_id" in seq_lookup.columns:
        seq_lookup["shap_sample_id"] = seq_lookup["shap_sample_id"].astype(int)

    for _, g in shap_db.groupby(key_cols + ["step_idx"], dropna=False):
        g_sorted = g.sort_values("abs_shap_value", ascending=False)
        first = g_sorted.iloc[0]

        seq_row = find_sequence_row(seq_lookup, key_cols, first)
        if seq_row is None:
            continue

        src_seq = parse_list_cell(seq_row["input_seq"])
        tgt_seq = parse_list_cell(seq_row["target_seq"])
        step_idx = int(first["step_idx"])

        if step_idx >= len(tgt_seq) or int(tgt_seq[step_idx]) == pad_id:
            continue

        valid_positions = [
            p for p in range(1, min(window_size, len(src_seq)) + 1)
            if src_seq[p - 1] not in {pad_id, bos_id}
        ]
        if len(valid_positions) < top_m:
            continue

        top_positions = g_sorted["src_position"].head(top_m).astype(int).tolist()
        random_positions = rng.choice(valid_positions, size=top_m, replace=False).astype(int).tolist()

        original = predict_one_step(model, tnsf, src_seq, tgt_seq, step_idx, top_k, window_size, bos_id, pad_id, device)
        top_pred = predict_one_step(
            model, tnsf,
            perturb_src_positions(src_seq, top_positions, mode, rng, pad_id, bos_id, seen_event_ids, background_src),
            tgt_seq, step_idx, top_k, window_size, bos_id, pad_id, device,
        )
        rand_pred = predict_one_step(
            model, tnsf,
            perturb_src_positions(src_seq, random_positions, mode, rng, pad_id, bos_id, seen_event_ids, background_src),
            tgt_seq, step_idx, top_k, window_size, bos_id, pad_id, device,
        )

        if original is None or top_pred is None or rand_pred is None:
            continue

        row = {
            "step_idx": step_idx,
            "top_m": int(top_m),
            "perturbation_mode": mode,
            "top_positions": json.dumps(top_positions),
            "random_positions": json.dumps(random_positions),
            "original_true_log_prob": original["true_log_prob"],
            "top_shap_true_log_prob": top_pred["true_log_prob"],
            "random_true_log_prob": rand_pred["true_log_prob"],
            "delta_true_log_prob_top_shap": top_pred["true_log_prob"] - original["true_log_prob"],
            "delta_true_log_prob_random": rand_pred["true_log_prob"] - original["true_log_prob"],
            "original_margin": original["topk_margin_log_prob"],
            "top_shap_margin": top_pred["topk_margin_log_prob"],
            "random_margin": rand_pred["topk_margin_log_prob"],
            "delta_margin_top_shap": top_pred["topk_margin_log_prob"] - original["topk_margin_log_prob"],
            "delta_margin_random": rand_pred["topk_margin_log_prob"] - original["topk_margin_log_prob"],
            "original_hit_top_k": original["hit_top_k"],
            "top_shap_hit_top_k": top_pred["hit_top_k"],
            "random_hit_top_k": rand_pred["hit_top_k"],
        }

        for col in key_cols:
            row[col] = first[col]
        if "stratum" in first.index:
            row["stratum"] = first["stratum"]

        cases.append(row)

        if max_cases is not None and len(cases) >= max_cases:
            break

    return pd.DataFrame(cases)


def find_sequence_row(seq_lookup, key_cols, first):
    """Find matching sequence metadata row."""
    mask = np.ones(len(seq_lookup), dtype=bool)

    for col in key_cols:
        mask = mask & (seq_lookup[col].astype(str) == str(first[col]))

    if not mask.any():
        return None

    return seq_lookup.loc[mask].iloc[0]


def summarize_ablation(ablation_df):
    """Summarize top-SHAP vs random perturbation results."""
    if ablation_df.empty:
        return pd.DataFrame()

    summary = {
        "n_cases": len(ablation_df),
        "mean_delta_true_log_prob_top_shap": ablation_df["delta_true_log_prob_top_shap"].mean(),
        "mean_delta_true_log_prob_random": ablation_df["delta_true_log_prob_random"].mean(),
        "median_delta_true_log_prob_top_shap": ablation_df["delta_true_log_prob_top_shap"].median(),
        "median_delta_true_log_prob_random": ablation_df["delta_true_log_prob_random"].median(),
        "mean_delta_margin_top_shap": ablation_df["delta_margin_top_shap"].mean(),
        "mean_delta_margin_random": ablation_df["delta_margin_random"].mean(),
        "top_shap_hit_change_rate": (
            ablation_df["original_hit_top_k"] != ablation_df["top_shap_hit_top_k"]
        ).mean(),
        "random_hit_change_rate": (
            ablation_df["original_hit_top_k"] != ablation_df["random_hit_top_k"]
        ).mean(),
    }

    return pd.DataFrame([summary]).round(6)