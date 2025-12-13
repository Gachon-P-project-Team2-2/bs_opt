# algos/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from generators import generate_synthetic_traffic, generate_random_mask

class History:
    """
    State의 히스토리를 기록하는 클래스
    """
    def __init__(self):
        self.records = []

    def add_record(self, state: State) -> None:
        self.records.append(state)

    def get_history(self) -> list[dict]:
        return copy.deepcopy(self.records)

    def get_best(self, **kwargs) -> dict:
        """
        히스토리에서 특정 키워드들에 대한 최적값을 반환
        """
        best_record = {}
        for key in kwargs:
            best_value = None
            for record in self.records:
                if key in record:
                    value = record[key]
                    if best_value is None or value < best_value:
                        best_value = value
            best_record[key] = best_value
        return best_record
    
    def get_worst(self, **kwargs) -> dict:
        """
        히스토리에서 특정 키워드들에 대한 최악값을 반환
        """
        worst_record = {}
        for key in kwargs:
            worst_value = None
            for record in self.records:
                if key in record:
                    value = record[key]
                    if worst_value is None or value > worst_value:
                        worst_value = value
            worst_record[key] = worst_value
        return worst_record
    
class State:
    """
    상태를 나타내는 구현 클래스
    """
    @classmethod
    def from_shape(
        cls,
        width: int,
        height: int,
        rng: np.random.Generator | None = None,
        traffic_params: dict | None = None,
        mask_params: dict | None = None,
    ) -> "State":
        """
        width/height만 주어졌을 때 랜덤 traffic/mask를 생성해 State를 만든다.
        """
        rng = rng or np.random.default_rng()
        traffic_pattern = (traffic_params or {}).get("pattern", "center_hotspot")
        traffic = generate_synthetic_traffic(width=width, height=height, rng=rng, pattern=traffic_pattern or None, params=traffic_params)
        block_prob = (mask_params or {}).get("block_prob", 0.1)
        mask = generate_random_mask(width, height, block_prob=block_prob, rng=rng)
        return cls(traffic_layer=traffic, mask_layer=mask, bs_layer=None, rng=rng)

    def __init__(
        self,
        traffic_layer: np.ndarray, # 2D array traffic demand map (required)
        mask_layer: np.ndarray, # 2D array not installable area mask (required)
        bs_layer: np.ndarray | None = None, # 2D array installed base stations (0/1 grid)
        rng: np.random.Generator | None = None,
    ):
        if traffic_layer is None or mask_layer is None:
            raise ValueError("traffic_layer and mask_layer are required.")

        self.rng = rng or np.random.default_rng()

        self.traffic_layer = np.asarray(traffic_layer, dtype=float)
        self.height, self.width = self.traffic_layer.shape

        self.mask_layer = np.asarray(mask_layer, dtype=bool)
        if self.mask_layer.shape != self.traffic_layer.shape:
            raise ValueError("mask_layer shape must match traffic_layer shape")

        if bs_layer is not None:
            bs = np.asarray(bs_layer, dtype=int)
            if bs.shape != self.traffic_layer.shape:
                raise ValueError("bs_layer shape must match traffic_layer shape")
            self.bs_layer = (bs > 0).astype(int)
        else:
            self.bs_layer = np.zeros_like(self.traffic_layer, dtype=int)
    
    def copy(self) -> "State":
        return State(
            traffic_layer=copy.deepcopy(self.traffic_layer),
            mask_layer=copy.deepcopy(self.mask_layer),
            bs_layer=copy.deepcopy(self.bs_layer),
            rng=self.rng,
        )

    def get_state(self) -> "State":
        """
        상태 반환 (deep copy)
        """
        return self.copy()
    
    def is_installable(self, x: int, y: int) -> bool: # x: column, y: row
        """
        설치 가능 여부 반환 (mask_layer가 True/1이면 설치 불가라고 가정)
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return not bool(self.mask_layer[y, x])

    def add_base_station(self, x: int, y: int) -> "State":
        """
        기지국 추가 (새로운 State 반환)
        """
        if not self.is_installable(x, y):
            return self.copy()
        if self.bs_layer[y, x] == 1:
            return self.copy()
        new_bs = self.bs_layer.copy()
        new_bs[y, x] = 1
        return State(self.traffic_layer, self.mask_layer, new_bs, self.rng)

    def remove_base_station(self, x: int, y: int) -> "State":
        """
        기지국 제거 (새로운 State 반환)
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return self.copy()
        if self.bs_layer[y, x] == 0:
            return self.copy()
        new_bs = self.bs_layer.copy()
        new_bs[y, x] = 0
        return State(self.traffic_layer, self.mask_layer, new_bs, self.rng)

    def move_base_station(self, src: tuple[int, int], dst: tuple[int, int]) -> "State":
        """
        src→dst 이동 (dst 설치 가능/빈 자리일 때), 새 State 반환
        """
        sx, sy = src
        dx, dy = dst
        if not (0 <= sx < self.width and 0 <= sy < self.height):
            return self.copy()
        if self.bs_layer[sy, sx] == 0:
            return self.copy()
        if not self.is_installable(dx, dy) or self.bs_layer[dy, dx] == 1:
            return self.copy()
        new_bs = self.bs_layer.copy()
        new_bs[sy, sx] = 0
        new_bs[dy, dx] = 1
        return State(self.traffic_layer, self.mask_layer, new_bs, self.rng)
    
    def random_state(self, n_bs: int, strategy: str = "uniform") -> "State":
        """
        랜덤하게 n_bs개 설치한 새 State 반환.
        strategy:
          - "uniform": 설치 가능 좌표에서 균등 무작위 선택 (기본)
          - "even": 맵 전역에 고르게 분포하도록 센터 격자 근처에서 선택
          - "gaussian": 하나의 중심을 잡고 가우시안 분포로 샘플링
          - "traffic_greedy": traffic가 높은 순으로 n_bs개 선택
          - "traffic_weighted": traffic 값을 확률로 사용해 선택 (비복원)
        """
        installable_coords = np.argwhere(self.mask_layer == 0)
        if installable_coords.shape[0] < n_bs:
            raise ValueError("Not enough installable cells for requested base stations")

        if strategy == "uniform":
            idxs = self.rng.choice(installable_coords.shape[0], size=n_bs, replace=False)
            coords = installable_coords[idxs]
        elif strategy == "traffic_greedy":
            flat_vals = self.traffic_layer[installable_coords[:, 0], installable_coords[:, 1]]
            order = np.argsort(flat_vals)[::-1]
            coords = installable_coords[order[:n_bs]]
        elif strategy == "traffic_weighted":
            flat_vals = self.traffic_layer[installable_coords[:, 0], installable_coords[:, 1]]
            weights = flat_vals.astype(float)
            if weights.sum() <= 0:
                idxs = self.rng.choice(installable_coords.shape[0], size=n_bs, replace=False)
            else:
                probs = weights / weights.sum()
                idxs = self.rng.choice(installable_coords.shape[0], size=n_bs, replace=False, p=probs)
            coords = installable_coords[idxs]
        elif strategy == "even":
            # 격자 센터를 균일 배치 후 가장 가까운 설치 가능 좌표를 선택
            side = int(np.ceil(np.sqrt(n_bs)))
            xs = np.linspace(0, self.width - 1, side)
            ys = np.linspace(0, self.height - 1, side)
            targets = np.array([(x, y) for y in ys for x in xs])[:n_bs]
            chosen = []
            available = installable_coords.tolist()
            for tx, ty in targets:
                if not available:
                    break
                av = np.array(available)
                d2 = (av[:, 1] - tx) ** 2 + (av[:, 0] - ty) ** 2
                idx = int(np.argmin(d2))
                chosen.append(av[idx])
                available.pop(idx)
            coords = np.array(chosen, dtype=int)
        elif strategy == "gaussian":
            center_idx = self.rng.integers(0, installable_coords.shape[0])
            cy, cx = installable_coords[center_idx]
            sigma = max(self.width, self.height) / 6.0
            coords = []
            attempts = 0
            while len(coords) < n_bs and attempts < n_bs * 20:
                gx = int(np.clip(self.rng.normal(loc=cx, scale=sigma), 0, self.width - 1))
                gy = int(np.clip(self.rng.normal(loc=cy, scale=sigma), 0, self.height - 1))
                attempts += 1
                if not self.is_installable(gx, gy):
                    continue
                if (gy, gx) in coords:
                    continue
                coords.append((gy, gx))
            if len(coords) < n_bs:
                # 부족하면 남은 수를 uniform으로 채움
                remaining = n_bs - len(coords)
                idxs = self.rng.choice(installable_coords.shape[0], size=remaining, replace=False)
                coords.extend([(int(installable_coords[i, 0]), int(installable_coords[i, 1])) for i in idxs])
            coords = np.array(coords, dtype=int)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        new_bs = np.zeros_like(self.bs_layer)
        for y, x in coords:
            new_bs[y, x] = 1
        return State(self.traffic_layer, self.mask_layer, new_bs, self.rng)
    
    def get_coordinates(self, bs_layer: np.ndarray=None) -> list[tuple[int, int]]:
        """
        기지국 좌표 리스트 반환
        """
        bs = bs_layer if bs_layer is not None else self.bs_layer
        coords = np.argwhere(bs > 0)
        return [(int(x), int(y)) for y, x in coords]
    
    def get_overall_map(self) -> np.ndarray:
        """
        3D array로 전체 맵 정보 반환 (traffic + mask + base stations)
        """
        traffic_layer = self.traffic_layer[:, :, None]
        mask_layer = self.mask_layer.astype(int)[:, :, None]
        bs_layer = self.bs_layer[:, :, None]
        overall_map = np.concatenate([traffic_layer, mask_layer, bs_layer], axis=2)
        return overall_map
        
    
class Objective(ABC):
    """
    모든 목적 함수가 상속받는 베이스 클래스
    """

    @abstractmethod
    def set_weights(self, **weights: Any) -> None:
        """
        목적 함수 가중치 설정 인터페이스
        """
        pass
    
    @abstractmethod
    def evaluate(self, state: "State") -> tuple[float, dict]:
        """
        주어진 state에 대한 비용(cost)과 메트릭 반환
        반환: (cost, metrics_dict)
          - cost: 최소화 대상
          - metrics:
              'coverage_ratio': 커버된 트래픽 / 전체 트래픽
              'num_bs': 기지국 수
        """
        raise NotImplementedError
    
class Algorithm:
    """
    모든 탐색 알고리즘이 상속받는 베이스 클래스
    """

    def __init__(
        self,
        objective: Objective,
        rng: np.random.Generator | None = None,
    ):
        self.objective = objective
        self.rng = rng or np.random.default_rng()

    def run(
        self,
        max_iter: int,
        init_state: np.ndarray | None = None,
        verbose: bool = False,
    ) -> History:
        """
        자식 클래스에서 구현해야 하는 알고리즘 실행 인터페이스
        """
        raise NotImplementedError
