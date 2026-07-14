#!/usr/bin/env python3
"""
Diagnostic: does additive Gaussian noise on the (z-scored) START time actually
destroy the per-client fingerprint, or is the picture still sharp?

Mechanism it models (exactly the pipeline's de-leak):
  * a client's underlying start = time[0] is the SAME scalar in train and test;
  * the train embeddings-gen run adds noise drawn from rng(0),
    the test run adds INDEPENDENT noise drawn from rng(1);
  * so the same client appears as   z+eps_train   and   z+eps_test.
If a nearest-neighbour matcher can still recover "which client is which" from
those two noisy views, the fingerprint survived. If recovery collapses to the
degeneracy ceiling (noise=0 row) / to 1/N, it is smeared out.

We evaluate recovery both in the raw noised z (1D) and in the FULL 17-dim block
that actually gets appended (raw z + Fourier bank + block z-standardization),
to make sure the high-frequency Fourier terms do not re-sharpen the identity.

Run on the server where the data lives, e.g.:
  python check_start_noise.py \
    --parquet /path/to/taobao/train \
    --time_name time \
    --grid 0 0.25 0.5 1.0 2.0
"""
import argparse
import numpy as np
import pandas as pd


def load_starts(parquet_path, time_name):
    s = pd.read_parquet(parquet_path, columns=[time_name])[time_name]
    return s.map(lambda a: float(a[0])).to_numpy()


def raw_time_block(z, freqs):
    z = np.asarray(z, dtype=np.float64)
    ang = np.outer(z, freqs)
    return np.concatenate([z[:, None], np.sin(ang), np.cos(ang)], axis=1)


def standardize(block):
    mu = block.mean(axis=0)
    sd = block.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (block - mu) / sd


def recover_1d(a_vals, b_vals, b_ids):
    """For each probe b, nearest gallery a; return fraction whose NN id == own id.
    Gallery ids are 0..N-1 (a is the full pool)."""
    order = np.argsort(a_vals)
    a_sorted = a_vals[order]
    pos = np.searchsorted(a_sorted, b_vals)
    pos = np.clip(pos, 1, len(a_sorted) - 1)
    left = a_sorted[pos - 1]
    right = a_sorted[pos]
    take_left = np.abs(b_vals - left) <= np.abs(b_vals - right)
    nn_pos = np.where(take_left, pos - 1, pos)
    nn_ids = order[nn_pos]
    return float(np.mean(nn_ids == b_ids))


def recover_nd(a, b, b_ids, chunk=200):
    """Euclidean NN recovery in R^d. a: (N,d) gallery (ids 0..N-1), b: (M,d) probes."""
    correct = 0
    for s in range(0, len(b), chunk):
        bb = b[s:s + chunk]                       # (c,d)
        d2 = ((a[None, :, :] - bb[:, None, :]) ** 2).sum(-1)  # (c,N)
        nn = np.argmin(d2, axis=1)
        correct += int(np.sum(nn == b_ids[s:s + chunk]))
    return correct / len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--time_name", default="time")
    ap.add_argument("--n_fourier", type=int, default=8)
    ap.add_argument("--grid", type=float, nargs="+", default=[0, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--gallery", type=int, default=20000, help="max clients in gallery")
    ap.add_argument("--probes", type=int, default=2000)
    ap.add_argument("--png", default="start_noise_check.png")
    args = ap.parse_args()

    starts = load_starts(args.parquet, args.time_name)
    n = len(starts)
    uniq = np.unique(starts)
    mean, std = float(starts.mean()), float(starts.std()) or 1.0
    z_all = (starts - mean) / std

    print(f"clients:            {n}")
    print(f"unique start values:{len(uniq)}  ({len(uniq)/n:.4%} of clients)")
    print(f"start mean/std:     {mean:.4g} / {std:.4g}")
    # largest collision group: how many clients share the single most common start
    vals, counts = np.unique(starts, return_counts=True)
    print(f"largest collision:  {counts.max()} clients share one start value")
    print(f"absolute chance (1/gallery): {1/min(n, args.gallery):.4%}")
    print()

    # fixed gallery subsample so all noise rows are comparable
    grng = np.random.default_rng(777)
    gal_idx = grng.choice(n, size=min(n, args.gallery), replace=False)
    z = z_all[gal_idx]
    N = len(z)
    probe_local = grng.choice(N, size=min(N, args.probes), replace=False)

    freqs = (2.0 * np.pi) * (2.0 ** np.arange(args.n_fourier))
    dim = 1 + 2 * args.n_fourier

    print(f"Fourier bank freqs: {freqs.tolist()}   block dim = {dim}")
    print()
    hdr = f"{'noise':>6} | {'recover_z':>10} | {'recover_block':>13} | {'corr_z(views)':>13}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for noise in args.grid:
        r_tr = np.random.default_rng(0)   # train-view noise
        r_te = np.random.default_rng(1)   # test-view  noise (independent)
        za = z + (r_tr.normal(0, noise, N) if noise > 0 else 0.0)  # gallery view
        zb_full = z + (r_te.normal(0, noise, N) if noise > 0 else 0.0)
        zb = zb_full[probe_local]
        b_ids = probe_local

        rec_z = recover_1d(za, zb, b_ids)

        # full appended block: build from each view, standardize by gallery-view stats
        Ba = raw_time_block(za, freqs)
        Bb = raw_time_block(zb_full, freqs)
        mu = Ba.mean(0); sd = Ba.std(0); sd = np.where(sd < 1e-6, 1.0, sd)
        Ba = ((Ba - mu) / sd).astype(np.float32)
        Bb = ((Bb - mu) / sd).astype(np.float32)
        rec_b = recover_nd(Ba, Bb[probe_local], b_ids)

        corr = float(np.corrcoef(za, zb_full)[0, 1])
        rows.append((noise, rec_z, rec_b, corr))
        print(f"{noise:>6.3g} | {rec_z:>10.4%} | {rec_b:>13.4%} | {corr:>13.4f}")

    print()
    ceil_z = rows[0][1]
    ceil_b = rows[0][2]
    print(f"noise=0 ceiling (degeneracy limit): recover_z={ceil_z:.4%}  recover_block={ceil_b:.4%}")
    print("Interpretation:")
    print("  * recover_* at your operating noise close to noise=0 ceiling  -> fingerprint SURVIVES (sharp).")
    print("  * recover_* collapsed toward 1/gallery and corr_z near 0      -> fingerprint SMEARED (good).")
    print("  * recover_block NOT higher than recover_z                     -> Fourier bank does not re-sharpen.")

    # optional picture
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        op = args.grid[min(len(args.grid) - 1, args.grid.index(1.0) if 1.0 in args.grid else -1)]
        r = np.random.default_rng(0)
        z_noisy = z + r.normal(0, op, N)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(z, bins=80, alpha=.6, label="z (clean)")
        ax[0].hist(z_noisy, bins=80, alpha=.6, label=f"z + noise({op})")
        ax[0].legend(); ax[0].set_title("start distribution")
        r2 = np.random.default_rng(1)
        sub = grng.choice(N, size=min(N, 4000), replace=False)
        ax[1].scatter(z[sub] + r.normal(0, op, len(sub)),
                      z[sub] + r2.normal(0, op, len(sub)), s=3, alpha=.3)
        ax[1].set_xlabel("train view"); ax[1].set_ylabel("test view")
        ax[1].set_title(f"same client, two noise draws (noise={op})")
        fig.tight_layout(); fig.savefig(args.png, dpi=110)
        print(f"\nsaved figure -> {args.png}")
    except Exception as e:
        print(f"\n(no figure: {e})")


if __name__ == "__main__":
    main()
