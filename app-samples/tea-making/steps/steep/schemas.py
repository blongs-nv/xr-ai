# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""steep: what it runs with."""

from __future__ import annotations

from xr_ai_kit.sop import WaitInputs


class SteepInputs(WaitInputs):
    """What steep runs with: duration_s, and {minutes} for its words."""

    @property
    def minutes(self) -> float:
        return self.duration_s / 60
