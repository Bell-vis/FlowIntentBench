"""Verified geometry of a consistently oriented, closed triangle component."""
from __future__ import annotations

import numpy as np


def closed_triangle_geometry(points, triangles, query_point=None):
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("a nonempty triangle mesh is required")
    vertices = points[triangles]
    if not np.isfinite(vertices).all():
        raise ValueError("surface coordinates must be finite")
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    canonical = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
    balance = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    closed = bool(np.all(counts == 2) and np.all(balance == 0))
    result = {"closed_consistently_oriented": closed}
    if query_point is not None:
        q = np.asarray(query_point, dtype=float)
        if q.shape != (3,) or not np.isfinite(q).all():
            raise ValueError("containment query must be a finite 3D point")
        result["query_point"] = q.tolist()
    if not closed:
        if query_point is not None:
            result["containment_status"] = "UNDEFINED_OPEN_SURFACE"
        return result
    # Shift the integration origin near the mesh for numerical stability.
    origin = vertices.reshape(-1, 3).mean(axis=0)
    a, b, c = (vertices[:, i] - origin for i in range(3))
    volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    signed_volume = float(volumes.sum())
    scale = float(np.linalg.norm(np.ptp(vertices.reshape(-1, 3), axis=0)))
    if abs(signed_volume) <= np.finfo(float).eps * max(scale ** 3, np.finfo(float).tiny) * 100:
        raise ValueError("closed surface has numerically zero enclosed volume")
    centroid = origin + (volumes[:, None] * (a + b + c) / 4.0).sum(axis=0) / signed_volume
    result.update(enclosed_volume=abs(signed_volume), enclosed_volume_centroid=centroid.tolist())
    if query_point is not None:
        # Oriented solid angle distinguishes containment from bounding-box overlap.
        q = np.asarray(query_point, dtype=float)
        a, b, c = (vertices[:, i] - q for i in range(3))
        na, nb, nc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (na * nb * nc + np.einsum("ij,ij->i", a, b) * nc
                       + np.einsum("ij,ij->i", b, c) * na
                       + np.einsum("ij,ij->i", c, a) * nb)
        winding = float(np.arctan2(numerator, denominator).sum() / (2 * np.pi))
        result.update(query_point=q.tolist(), absolute_winding_number=abs(winding),
                      encloses_query_point=bool(abs(winding) > 0.5))
    return result
