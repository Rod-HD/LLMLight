"""Seed Manager (Component 10).

Đặt random seed cho mọi nguồn ngẫu nhiên trong pipeline:

* ``random`` (Python stdlib)
* ``numpy.random`` (nếu numpy được cài)
* ``torch.manual_seed`` + ``torch.cuda.manual_seed_all`` (nếu torch được cài
  và CUDA khả dụng)
* ``os.environ["PYTHONHASHSEED"]``

CityFlow simulator KHÔNG được seed ở đây — seed phải truyền trực tiếp qua
``CityFlowEngine.__init__(seed=...)`` (do design đã quyết định tách concern).

Mọi runner script PHẢI gọi ``SeedManager().apply(seed)`` đầu tiên trước khi
khởi tạo bất kỳ thành phần ngẫu nhiên nào (bao gồm ``CityFlowEngine``,
``LightGPTInference``, ``MultiBackendAPIClient``, training loops).

Pattern điển hình trong runner::

    sm = SeedManager()                     # đọc RANDOM_SEED từ env
    seed = sm.seed_for_run(run_id)         # base + run_id
    sm.apply(seed)                          # set tất cả nguồn ngẫu nhiên
    engine = CityFlowEngine(config_path, seed=seed)  # CityFlow seed riêng

Module import sạch ngay cả khi ``numpy`` / ``torch`` chưa được cài (môi
trường dev Windows). Khi gọi ``apply()``, các backend bị thiếu được skip
một cách im lặng (chỉ log debug) — KHÔNG raise. Ở runtime trong WSL2 venv,
cả numpy và torch đều phải tồn tại, nên ``apply()`` sẽ seed đầy đủ.

_Requirements_: 1.9, 8.9, 12.8
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional backend imports — guarded so SeedManager module imports even khi
# numpy/torch chưa được cài (môi trường dev Windows trước khi setup_env.sh
# chạy trong WSL2). Ở runtime production (WSL2 venv), cả hai đều tồn tại.
# ---------------------------------------------------------------------------

def _lazy_numpy() -> Any | None:
    try:
        import numpy as _numpy  # type: ignore[import-not-found]
        return _numpy
    except Exception as exc:  # pragma: no cover
        logger.debug("numpy not importable: %s", exc)
        return None


def _lazy_torch() -> Any | None:
    # Lazy import: avoid triggering CUDA init at module import time.
    # torch spawns a GPU polling thread on import when CUDA is present,
    # which hangs indefinitely in WSL2 via /dev/dxg poll().
    # Skip entirely when LLMLIGHT_NO_TORCH_SEED is set (Phase 2 API runs).
    if os.environ.get("LLMLIGHT_NO_TORCH_SEED"):
        return None
    try:
        import torch as _torch  # type: ignore[import-not-found]
        return _torch
    except Exception as exc:  # pragma: no cover
        logger.debug("torch not importable: %s", exc)
        return None


class SeedManager:
    """Quản lý random seed cho toàn bộ pipeline.

    Attributes:
        base_seed: Seed gốc đọc từ ``RANDOM_SEED`` env var (mặc định 42)
            hoặc truyền vào constructor.
    """

    DEFAULT_SEED: int = 42
    ENV_VAR: str = "RANDOM_SEED"

    def __init__(self, base_seed: int | None = None) -> None:
        """Khởi tạo SeedManager.

        Args:
            base_seed: Nếu truyền vào, dùng làm base seed. Nếu ``None``,
                đọc từ env ``RANDOM_SEED``; nếu env không set hoặc không
                phải số hợp lệ, fallback về ``DEFAULT_SEED`` (42).
        """
        if base_seed is not None:
            self.base_seed = int(base_seed)
        else:
            self.base_seed = self._read_env_seed()

    @classmethod
    def _read_env_seed(cls) -> int:
        """Đọc ``RANDOM_SEED`` từ env, fallback ``DEFAULT_SEED`` nếu thiếu /
        không parse được."""
        raw = os.environ.get(cls.ENV_VAR)
        if raw is None or raw.strip() == "":
            return cls.DEFAULT_SEED
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Env %s=%r is not an integer; falling back to default %d.",
                cls.ENV_VAR,
                raw,
                cls.DEFAULT_SEED,
            )
            return cls.DEFAULT_SEED

    def seed_for_run(self, run_id: int) -> int:
        """Tính seed cho run thứ ``run_id``.

        Spec yêu cầu seed = ``base_seed + run_id`` để các run trong cùng
        một experiment là tất định nhưng khác biệt
        (Requirement 8.9, 12.8).

        Args:
            run_id: Index của run (0, 1, 2, ...).

        Returns:
            Seed tuyệt đối để truyền vào ``apply()`` và
            ``CityFlowEngine(seed=...)``.
        """
        return self.base_seed + int(run_id)

    def apply(self, seed: int) -> None:
        """Áp dụng ``seed`` vào MỌI nguồn ngẫu nhiên có sẵn.

        Thứ tự áp dụng:
          1. ``os.environ["PYTHONHASHSEED"]`` (ảnh hưởng hash randomization
             cho process con — ghi trước random/numpy/torch để đồng bộ).
          2. ``random.seed`` (Python stdlib).
          3. ``numpy.random.seed`` (nếu numpy import được).
          4. ``torch.manual_seed`` + ``torch.cuda.manual_seed_all`` (nếu
             torch import được; cuda backend chỉ chạy khi
             ``torch.cuda.is_available()``).

        Backends bị thiếu được skip một cách im lặng (chỉ log debug) —
        KHÔNG raise. Điều này cho phép module được import + test trên môi
        trường dev Windows nơi torch chưa được cài, đồng thời vẫn seed đầy
        đủ trong WSL2 venv.

        Args:
            seed: Giá trị seed (thường lấy từ ``seed_for_run(run_id)``).
        """
        seed = int(seed)

        # 1) PYTHONHASHSEED — set trước để các tiến trình con kế thừa.
        os.environ["PYTHONHASHSEED"] = str(seed)

        # 2) Python random
        random.seed(seed)

        # 3) NumPy
        _numpy = _lazy_numpy()
        if _numpy is not None:
            try:
                _numpy.random.seed(seed)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to seed numpy.random: %s", exc)
        else:
            logger.debug("numpy not available; skipping numpy seed.")

        # 4) PyTorch (CPU + CUDA) — lazy import to avoid CUDA init at module load
        _torch = _lazy_torch()
        if _torch is not None:
            try:
                _torch.manual_seed(seed)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to seed torch: %s", exc)

            try:
                if _torch.cuda.is_available():
                    _torch.cuda.manual_seed_all(seed)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "torch.cuda check failed (%s); skipping CUDA seed.",
                    exc,
                )
        else:
            logger.debug("torch not available; skipping torch/cuda seed.")

        logger.info("SeedManager.apply: seeded all backends with seed=%d.", seed)
