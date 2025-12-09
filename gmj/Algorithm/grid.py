import numpy as np

def _normalize_and_scale(traffic, params):
    """Apply optional scaling/normalization."""
    traffic = traffic.astype(float)

    # Optional pre-normalization scaling to a target sum or mean
    target_sum = params.get("target_sum")
    target_mean = params.get("target_mean")
    eps = 1e-8
    if target_sum is not None and traffic.sum() > eps:
        traffic = traffic * (float(target_sum) / (traffic.sum() + eps))
    if target_mean is not None and traffic.mean() > eps:
        traffic = traffic * (float(target_mean) / (traffic.mean() + eps))

    # Normalize to [0, 1] unless disabled
    if params.get("normalize", True):
        traffic = (traffic - traffic.min()) / (traffic.max() - traffic.min() + eps)

    if params.get("clip_0_1", True):
        traffic = np.clip(traffic, 0.0, 1.0)
    return traffic


def _add_noise(traffic, rng, params):
    noise_std = float(params.get("noise_std", 0.05))
    if noise_std <= 0:
        return traffic
    noise = rng.normal(loc=0.0, scale=noise_std, size=traffic.shape)
    return traffic + noise


def generate_synthetic_traffic(
    width,
    height,
    pattern="center_hotspot",
    rng=None,
    params=None,
):
    """
    width x height 격자에 대한 트래픽 맵을 생성한다.
    """
    if rng is None:
        rng = np.random.default_rng()
    params = params or {}

    y, x = np.mgrid[0:height, 0:width]
    traffic = np.zeros((height, width), dtype=float)

    if pattern == "random":
        traffic = rng.random((height, width))
        traffic = _add_noise(traffic, rng, params)
        return _normalize_and_scale(traffic, params)

    if pattern == "center_hotspot":
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        sigma_x = params.get("sigma_x", width / 4.0)
        sigma_y = params.get("sigma_y", height / 4.0)
        gauss = np.exp(
            -(((x - cx) ** 2) / (2 * sigma_x**2) + ((y - cy) ** 2) / (2 * sigma_y**2))
        )
        traffic = gauss
        noise_params = dict(params)
        noise_params["noise_std"] = params.get("noise_std", 0.2)
        traffic = _add_noise(traffic, rng, noise_params)
        return _normalize_and_scale(traffic, params)

    if pattern == "multi_hotspot":
        centers = params.get("centers")
        n_centers = params.get("n_centers", 3)
        sigma_x = params.get("sigma_x", width / 6.0)
        sigma_y = params.get("sigma_y", height / 6.0)
        if centers is None:
            centers = [
                (rng.uniform(0, width - 1), rng.uniform(0, height - 1))
                for _ in range(n_centers)
            ]
        traffic = np.zeros_like(x, dtype=float)
        for cx, cy in centers:
            traffic += np.exp(
                -(((x - cx) ** 2) / (2 * sigma_x**2) + ((y - cy) ** 2) / (2 * sigma_y**2))
            )
        traffic = _add_noise(traffic, rng, params)
        return _normalize_and_scale(traffic, params)

    if pattern == "ring":
        center = params.get("center", ((width - 1) / 2.0, (height - 1) / 2.0))
        radius = float(params.get("radius", min(width, height) / 3.0))
        thickness = float(params.get("thickness", radius / 4.0))
        cx, cy = center
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        traffic = np.exp(-((r - radius) ** 2) / (2 * thickness**2))
        traffic = _add_noise(traffic, rng, params)
        return _normalize_and_scale(traffic, params)

    if pattern == "gradient":
        direction = params.get("direction", "ew")  # ew: east→west increase
        if direction == "ew":
            base = np.linspace(0, 1, width)[None, :]
            traffic = np.repeat(base, height, axis=0)
        elif direction == "ns":
            base = np.linspace(0, 1, height)[:, None]
            traffic = np.repeat(base, width, axis=1)
        else:
            raise ValueError(f"Unknown gradient direction: {direction}")
        traffic = _add_noise(traffic, rng, params)
        return _normalize_and_scale(traffic, params)

    if pattern == "stripe":
        orientation = params.get("orientation", "vertical")
        stripe_pos = params.get("stripe_pos", width // 2 if orientation == "vertical" else height // 2)
        stripe_width = params.get("stripe_width", 2)
        decay = params.get("decay", stripe_width)
        if orientation == "vertical":
            dist = np.abs(x - stripe_pos)
        elif orientation == "horizontal":
            dist = np.abs(y - stripe_pos)
        else:
            raise ValueError(f"Unknown stripe orientation: {orientation}")
        traffic = np.exp(-(dist**2) / (2 * (decay**2)))
        traffic = _add_noise(traffic, rng, params)
        return _normalize_and_scale(traffic, params)

    if pattern == "checkerboard":
        block = params.get("block", 2)
        high = params.get("high", 1.0)
        low = params.get("low", 0.2)
        traffic = ((x // block + y // block) % 2).astype(float)
        traffic = traffic * (high - low) + low
        traffic = _add_noise(traffic, rng, params)
        return _normalize_and_scale(traffic, params)

    if pattern == "random_clusters":
        n_clusters = params.get("n_clusters", rng.poisson(5))
        sigma = params.get("sigma", min(width, height) / 8.0)
        traffic = np.zeros_like(x, dtype=float)
        for _ in range(max(1, n_clusters)):
            cx = rng.uniform(0, width - 1)
            cy = rng.uniform(0, height - 1)
            traffic += np.exp(-(((x - cx) ** 2) + ((y - cy) ** 2)) / (2 * sigma**2))
        traffic = _add_noise(traffic, rng, params)
        return _normalize_and_scale(traffic, params)

    raise ValueError(f"Unknown pattern: {pattern}")
