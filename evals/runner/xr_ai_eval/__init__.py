# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One runner and result format for every xr-ai eval.

See ``evals/EVAL_SCHEMA.md`` for the case and result contracts and ``evals/README.md`` for usage.
"""

from xr_ai_eval.adapters import Adapter, AdapterContext, find_adapter
from xr_ai_eval.cases import Case, CaseError, Suite, Turn, load_suite
from xr_ai_eval.profiles import Endpoint, Profile, ProfileError, load_profile
from xr_ai_eval.schema import SCHEMA_VERSION, Record, SchemaError, read_records

__all__ = [
    "SCHEMA_VERSION",
    "Adapter",
    "AdapterContext",
    "Case",
    "CaseError",
    "Endpoint",
    "Profile",
    "ProfileError",
    "Record",
    "SchemaError",
    "Suite",
    "Turn",
    "find_adapter",
    "load_profile",
    "load_suite",
    "read_records",
]
