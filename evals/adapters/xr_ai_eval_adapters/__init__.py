# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters that call the existing eval harnesses and convert their output to ``xr_ai_eval`` records.

The harnesses are not modified. Each adapter runs a harness as a subprocess in its own project
environment, then parses what the harness printed or wrote. See ``evals/README.md``.
"""

from xr_ai_eval_adapters.base import HarnessAdapter, Parsed, Step
from xr_ai_eval_adapters.harnesses import ADAPTERS

__all__ = ["ADAPTERS", "HarnessAdapter", "Parsed", "Step"]
