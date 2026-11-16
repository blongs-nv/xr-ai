# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private core of ``xr_ai_next``: the app, components, participants, task scopes, enrollment, model roles, frames,
the hub adapter, tracing, the virtual clock and the test kit.

Nothing here is public. Apps and kits import the public names from ``xr_ai_next`` and its documented modules, which
import them from here (``App``, ``Component``, ``Participant``, ``Session`` and ``Turn`` are defined here, one class
each). The lifecycle and profile contracts these modules implement are ``docs/LIFECYCLE_CONTRACT.md`` and
``docs/PROFILES_SCHEMA.md`` in the ``xr-ai-next`` package.
"""
