# core/grid.py
from __future__ import annotations
import numpy as np


def _normalize_and_scale(
    traffic: np.ndarray,
    params: dict,
) -> np.ndarray:
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


def _add_noise(traffic: np.ndarray, rng: np.random.Generator, params: dict) -> np.ndarray:
    noise_std = float(params.get("noise_std", 0.05))
    if noise_std <= 0:
        return traffic
    noise = rng.normal(loc=0.0, scale=noise_std, size=traffic.shape)
    return traffic + noise


def generate_synthetic_traffic(
    width: int,
    height: int,
    pattern: str = "center_hotspot",
    rng: np.random.Generator | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """
    width x height 격자에 대한 트래픽 맵을 생성한다.
    
    args:
      - width : 가로
      - height : 세로
      - pattern : 패턴 
      - rng : 난수 생성기
      - params : 패턴 생성 시 사용되는 하이퍼 파라미터
      
    returns: (height, width) shape의 2D numpy array

    pattern:
      - "random": 균일 랜덤
      - "center_hotspot": 중앙 가우시안 + 잡음
      - "multi_hotspot": 여러 개의 가우시안 합
      - "ring": 도넛 형태 (중심에서 특정 반경에 띠)
      - "gradient": 동→서 선형 증가 (direction 파라미터로 제어 가능)
      - "stripe": 도로/리버 같은 띠 모양
      - "checkerboard": 블록형 체커 패턴
      - "random_clusters": 랜덤 클러스터(포아송) + 커널 합산

    params:
      - noise_std: float, 가우시안 노이즈 표준편차 (기본 0.05)
      - normalize: bool, 0~1 정규화 여부 (기본 True)
      - clip_0_1: bool, 정규화 후 클리핑 여부 (기본 True)
      - target_sum / target_mean: 스케일 조정용
      - pattern별 세부: see 각 분기
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

if __name__ == "__main__":
    # quick visualization test (saved to disk instead of plt.show for headless use)
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    demo_patterns = [
        ("random", {}),
        ("center_hotspot", {}),
        ("multi_hotspot", {"n_centers": 4}),
        ("ring", {"radius": 8, "thickness": 2}),
        ("gradient", {"direction": "ew"}),
        ("stripe", {"orientation": "vertical", "stripe_width": 3}),
        ("checkerboard", {"block": 3}),
        ("random_clusters", {"n_clusters": 6, "sigma": 3.0}),
    ]
    for name, p in demo_patterns:
        traffic_map = generate_synthetic_traffic(50, 50, pattern=name, rng=rng, params=p)
        plt.figure(figsize=(16, 16))
        plt.imshow(traffic_map, origin="lower")
        plt.title(f"pattern={name}")
        plt.colorbar(label="Traffic")
        plt.tight_layout()
        plt.savefig(f"../tests/outputs/traffic_sample_{name}.png")
        plt.close()
