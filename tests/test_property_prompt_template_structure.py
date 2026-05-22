"""Property-Based Test — Property 3: Prompt Template Structure.

**Validates: Requirements 3.3, 12.3**

Cho mọi state dict hợp lệ (theo contract của
:class:`src.observation_parser.ObservationParser`),
``parser.parse(state)`` SHALL trả về prompt text chứa đủ 3 section nhận
diện được:

- ``## Observation``
- ``## Instruction``
- ``## Output Format``

Ngoài ra:

- Ba header phải xuất hiện THEO THỨ TỰ
  ``Observation < Instruction < Output Format`` (Requirement 3 AC 3 yêu
  cầu prompt được phân tách rõ ràng và section ``output format`` mô tả
  format trả về của LLM, nên nó nằm sau cùng).
- Output phải đề cập tới ``<signal>`` (output spec phải hướng dẫn LLM
  trả về pha tín hiệu trong tag ``<signal>...</signal>`` — Requirement
  3 AC 3.c, AC 4.1).

Strategy ``valid_state_dict_strategy`` được thiết kế **giống** strategy
sẽ dùng cho Property 2 (Observation Parser Determinism, Task 3.2): sinh
state dict với:

- ``lane_vehicle_count``: dict[str, int] với 1-12 lane (lane_id là
  ASCII printable không trống), queue length là ``int >= 0`` (không có
  giới hạn trên cố định — Requirement 3 AC 4).
- ``current_phase``: phải thuộc :data:`src.sim_config.VALID_PHASES`.
- ``current_phase_time``: ``int >= 0``.

Sử dụng ``@settings(max_examples=100, deadline=None)`` để bảo đảm tối
thiểu 100 iterations theo yêu cầu của Task 3.3 và tránh false-failure
do deadline trên Windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.observation_parser import ObservationParser  # noqa: E402
from src.sim_config import VALID_PHASES  # noqa: E402


# -------------------------------------------------------------------------
# Hypothesis strategies
# -------------------------------------------------------------------------

# Lane id: ASCII non-empty string. We restrict to alphanumerics + a few
# common separators ('_', '-') to keep failing examples readable; the
# parser itself only requires non-empty ``str``.
_lane_id_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="_-",
    ),
    min_size=1,
    max_size=20,
)

# Queue length: int >= 0, no fixed upper bound. We cap at 10_000 just to
# keep generated dicts small, but the parser itself imposes no upper
# bound (Requirement 3 AC 4).
_queue_length_strategy = st.integers(min_value=0, max_value=10_000)

# Lane vehicle count map: between 1 and 12 lanes (matches roadnet sizes
# in Jinan/Hangzhou — typical 4-arm intersection has 12 incoming lanes).
_lane_vehicle_count_strategy = st.dictionaries(
    keys=_lane_id_strategy,
    values=_queue_length_strategy,
    min_size=1,
    max_size=12,
)

# Current phase: one of the four valid phases.
_current_phase_strategy = st.sampled_from(VALID_PHASES)

# Current phase time: int >= 0, capped to a realistic range
# (one full episode at Full_mode == 3600 seconds).
_current_phase_time_strategy = st.integers(min_value=0, max_value=3600)


@st.composite
def valid_state_dict_strategy(draw: st.DrawFn) -> dict:
    """Generate a valid state dict that ``ObservationParser.parse`` accepts."""
    return {
        "lane_vehicle_count": draw(_lane_vehicle_count_strategy),
        "current_phase": draw(_current_phase_strategy),
        "current_phase_time": draw(_current_phase_time_strategy),
    }


# -------------------------------------------------------------------------
# Property 3: Prompt Template Structure
# -------------------------------------------------------------------------


@given(state=valid_state_dict_strategy())
@settings(max_examples=100, deadline=None)
def test_prompt_contains_three_section_headers_in_order(state: dict) -> None:
    """**Validates: Requirements 3.3, 12.3**

    Cho mọi state hợp lệ, ``parse(state)``:

    1. Chứa cả 3 header ``## Observation`` / ``## Instruction`` /
       ``## Output Format``.
    2. Ba header xuất hiện theo thứ tự
       ``Observation < Instruction < Output Format``.
    3. Đề cập tới ``<signal>`` (output spec hướng dẫn LLM trả về pha
       trong tag ``<signal>...</signal>``).
    """
    parser = ObservationParser()

    prompt = parser.parse(state)

    # 1. Cả 3 section header phải xuất hiện.
    assert "## Observation" in prompt, (
        "Prompt missing '## Observation' header"
    )
    assert "## Instruction" in prompt, (
        "Prompt missing '## Instruction' header"
    )
    assert "## Output Format" in prompt, (
        "Prompt missing '## Output Format' header"
    )

    # 2. Thứ tự xuất hiện: Observation < Instruction < Output Format.
    idx_obs = prompt.index("## Observation")
    idx_ins = prompt.index("## Instruction")
    idx_out = prompt.index("## Output Format")
    assert idx_obs < idx_ins < idx_out, (
        "Section headers out of order: "
        f"Observation@{idx_obs}, Instruction@{idx_ins}, "
        f"Output Format@{idx_out}"
    )

    # 3. Output spec phải đề cập tới <signal> tag để hướng dẫn LLM.
    assert "<signal>" in prompt, (
        "Prompt does not mention '<signal>' tag in the output spec"
    )
