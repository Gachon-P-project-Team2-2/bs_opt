from __future__ import annotations
import numpy as np
from base import Objective, State

class CoverageObjective(Objective):
    def __init__(
        self,
        *,
        pixel_to_km: float = 0.06,   # grid_resolution
        k_scale: float = 1e8,        # K
        beta: float = 4.5,           # g
        traffic_scale: float = 3.0,  # traffic_2d *= 3.0
    ):
        self.pixel_to_km = pixel_to_km
        self.k_scale = k_scale
        self.beta = beta
        self.traffic_scale = traffic_scale

        self.A = 50.0 # 기준 거리에서 평균 손실
        self.B = 40.0 # 거리 로그 스케일링 계수
        self.P_threshold = 100.0 # 수신 임계 손실 (dB)

        # lazy init
        self.is_init = False
        self.traffic_map = None
        self.total_traffic = None
        self.h = None
        self.w = None
        self.x_grid = None
        self.y_grid = None

        # 좌표별 커버 마스크 캐시
        self.mask_cache: dict[tuple[int, int], np.ndarray] = {}

    def _lazy_init(self, state: State):
        """
        code1과 동일하게:
        - traffic_map을 사용하고(여기서는 state.traffic_layer),
        - traffic_scale(기본 3.0) 적용,
        - 그리드 생성
        """
        self.traffic_map = state.traffic_layer.astype(float) * self.traffic_scale
        self.total_traffic = float(np.sum(self.traffic_map))

        h, w = self.traffic_map.shape
        self.h, self.w = h, w

        # code1과 동일한 (row, col) 계산을 벡터화하기 위한 grid
        # 주의: code1에서 row = px // side_len, col = px % side_len
        # 여기서는 y_grid=row, x_grid=col로 맞추면 혼동이 덜합니다.
        self.y_grid, self.x_grid = np.mgrid[0:h, 0:w]

        self.is_init = True

    def _get_coverage_mask(self, x: int, y: int) -> np.ndarray:
        """
        기지국 하나가 커버하는 셀 마스크 반환
        - 픽셀 트래픽 값에 따라 (>=3.0:0.3, >=1.5:0.6, else:1.2)
        - N ~ Normal(0, sqrt(10)) 를 (기지국,픽셀)마다 독립으로 생성
          => 여기서는 "기지국 좌표마다 noise_map을 새로 생성"하여 고정시킴(코드1의 사전계산과 동일 효과)
        - P_loss < P_threshold AND dist_km <= max_radius 이면 True
        """
        # 캐시 확인
        coord = (x, y)
        if coord in self.mask_cache:
            return self.mask_cache[coord]

        # 거리(km) 계산
        dist_grid = np.sqrt((self.y_grid - x) ** 2 + (self.x_grid - y) ** 2)
        dist_km = dist_grid * self.pixel_to_km
        dist_km = np.maximum(dist_km, 0.001)

        # 픽셀 트래픽 값으로 max_radius 결정
        max_radius = np.where(
            self.traffic_map >= 3.0, 0.3,
            np.where(self.traffic_map >= 1.5, 0.6, 1.2)
        )

        # coord마다 한 장의 noise_map을 만들어 고정
        N = np.random.normal(0.0, np.sqrt(10.0), size=(self.h, self.w))

        P_loss = self.A + self.B * np.log10(dist_km) + N

        mask = (dist_km <= max_radius) & (P_loss < self.P_threshold)
        self.mask_cache[coord] = mask
        return mask

    def evaluate(self, state: State):
        if not self.is_init:
            self._lazy_init(state)

        coords = state.get_coordinates()
        num_bs = len(coords)

        # code1: n_bts==0 이면 inf
        if num_bs == 0:
            return float("inf"), {
                "coverage_ratio": 0.0,
                "coverage_percent": 0.0,
                "num_bs": 0,
            }

        masks = []
        for c in coords:
            bx = int(c[0])
            by = int(c[1])
            masks.append(self._get_coverage_mask(bx, by))

        # 여러 기지국 커버 합집합
        final_mask = np.logical_or.reduce(masks) if masks else np.zeros((self.h, self.w), dtype=bool)

        covered_traffic = float(np.sum(self.traffic_map * final_mask))
        total_traffic = self.total_traffic
        
        coverage_percent = (covered_traffic / total_traffic) * 100.0 if total_traffic > 0 else 0.0
        coverage_ratio = (covered_traffic / total_traffic) if total_traffic > 0 else 0.0

        if coverage_percent == 0.0:
            cost = float("inf")
        else:
            cost = self.k_scale * (num_bs / (coverage_percent ** self.beta))

        return float(cost), {
            "coverage_ratio": float(coverage_ratio),
            "coverage_percent": float(coverage_percent),
            "num_bs": int(num_bs)
        }

    def set_weights(self, **weights):
        return super().set_weights(**weights)
    