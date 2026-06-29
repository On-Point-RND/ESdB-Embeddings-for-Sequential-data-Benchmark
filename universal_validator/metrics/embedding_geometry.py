from __future__ import annotations

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors


def effective_rank(x: np.ndarray) -> float:
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0, None)
    total = eigvals.sum()
    if total <= 0:
        return 0.0

    probs = eigvals[eigvals > 0] / total
    return float(np.exp(-(probs * np.log(probs)).sum()))


def rankme(x: np.ndarray) -> float:
    singular_values = np.linalg.svd(x, compute_uv=False)
    total = singular_values.sum()
    if total <= 0:
        return 0.0

    probs = singular_values[singular_values > 0] / total
    return float(np.exp(-(probs * np.log(probs)).sum()))


def stable_rank(x: np.ndarray) -> float:
    centered = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if len(singular_values) == 0 or singular_values[0] == 0:
        return 0.0
    return float((singular_values**2).sum() / singular_values[0] ** 2)


def anisotropy(x: np.ndarray) -> float:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = x / np.clip(norms, 1e-12, None)
    summed = normalized.sum(axis=0)
    n = normalized.shape[0]
    if n < 2:
        return 0.0

    pairwise_sum = float(summed @ summed - n)
    return pairwise_sum / (n * (n - 1))


def covariance_eigenvalues(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    return np.clip(np.linalg.eigvalsh(cov), 0.0, None)


def entropy_rank(values: np.ndarray, eps: float = 1e-12) -> float:
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    values = values[values > eps]
    total = values.sum()
    if total <= eps:
        return 0.0
    probs = values / total
    return float(np.exp(-(probs * np.log(probs)).sum()))


def effdim(x: np.ndarray) -> float:
    eigvals = covariance_eigenvalues(x)
    total = eigvals.sum()
    sq_total = np.square(eigvals).sum()
    if total <= 0 or sq_total <= 0:
        return 0.0
    return float(total**2 / sq_total)


def total_compression(source: np.ndarray, target: np.ndarray) -> float:
    source_effdim = effdim(source)
    target_effdim = effdim(target)
    if source_effdim <= 0 or target_effdim <= 0:
        return float("nan")
    return float(np.log(target_effdim / source_effdim))


def pca_filter_whiten(
    x: np.ndarray,
    sigma: float,
    eps: float = 1e-12,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = eigvals > sigma
    if not np.any(keep):
        return np.empty((x.shape[0], 0), dtype=np.float64)
    return centered @ eigvecs[:, keep] / np.sqrt(np.clip(eigvals[keep], eps, None))


def ksg_mutual_information(u: np.ndarray, v: np.ndarray, k: int) -> float:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    n = u.shape[0]
    if n <= k + 1:
        return 0.0
    joint = np.concatenate([u, v], axis=1)
    distances, _ = (
        NearestNeighbors(metric="chebyshev", n_neighbors=k + 1)
        .fit(joint)
        .kneighbors(joint)
    )
    radii = np.nextafter(distances[:, k], 0.0)
    u_nn = NearestNeighbors(metric="chebyshev").fit(u)
    v_nn = NearestNeighbors(metric="chebyshev").fit(v)
    n_u = np.array(
        [
            len(idx) - 1
            for idx in u_nn.radius_neighbors(u, radius=radii, return_distance=False)
        ]
    )
    n_v = np.array(
        [
            len(idx) - 1
            for idx in v_nn.radius_neighbors(v, radius=radii, return_distance=False)
        ]
    )
    estimate = digamma(k) + digamma(n) - np.mean(digamma(n_u + 1) + digamma(n_v + 1))
    return float(max(0.0, estimate))


def asmi_ksg_pca_whiten(
    x: np.ndarray,
    sigma: float,
    projection_dim: int,
    n_projections: int,
    ksg_k: int,
    seed: int,
) -> float:
    x = pca_filter_whiten(x, sigma=sigma)
    dim = x.shape[1]
    k = min(projection_dim, dim)
    if k == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    noisy = x + rng.normal(scale=sigma, size=x.shape)
    values = []
    for _ in range(n_projections):
        q, _ = np.linalg.qr(rng.normal(size=(dim, k)))
        projection = q[:, :k]
        values.append(
            ksg_mutual_information(x @ projection, noisy @ projection, k=ksg_k)
        )
    return float(np.mean(values))


def lidar_from_views(view_embeddings: np.ndarray, reg: float = 1e-4) -> float:
    view_embeddings = np.asarray(view_embeddings, dtype=np.float64)
    n_samples, n_views, dim = view_embeddings.shape
    class_means = view_embeddings.mean(axis=1)
    centered_means = class_means - class_means.mean(axis=0, keepdims=True)
    between_cov = centered_means.T @ centered_means / max(n_samples - 1, 1)
    centered_views = view_embeddings - class_means[:, None, :]
    flat_centered = centered_views.reshape(n_samples * n_views, dim)
    within_cov = flat_centered.T @ flat_centered / max(n_samples * (n_views - 1), 1)
    reg_scale = np.trace(within_cov) / max(dim, 1)
    within_cov = within_cov + reg * max(reg_scale, 1e-12) * np.eye(dim)
    eigvals_w, eigvecs_w = np.linalg.eigh(within_cov)
    inv_sqrt_w = (
        eigvecs_w
        @ np.diag(1.0 / np.sqrt(np.clip(eigvals_w, 1e-12, None)))
        @ eigvecs_w.T
    )
    lidar_matrix = inv_sqrt_w @ between_cov @ inv_sqrt_w
    lidar_matrix = 0.5 * (lidar_matrix + lidar_matrix.T)
    return entropy_rank(np.linalg.eigvalsh(lidar_matrix))
