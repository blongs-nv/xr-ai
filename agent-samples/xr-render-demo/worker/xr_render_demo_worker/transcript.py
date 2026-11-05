# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist completed exchanges from both routed destinations."""

import time

from xr_ai_tools.text_memory import AddTranscriptRequest, TextMemoryTools


async def persist_exchange(
    memory: TextMemoryTools,
    *,
    participant_id: str,
    query: str,
    response: str,
    timestamp_us: int,
) -> None:
    await memory.add_transcript.execute(
        AddTranscriptRequest(
            source_id=f"{participant_id}:user",
            timestamp_us=timestamp_us or time.time_ns() // 1_000,
            text=query,
        )
    )
    await memory.add_transcript.execute(
        AddTranscriptRequest(
            source_id=f"{participant_id}:agent",
            timestamp_us=time.time_ns() // 1_000,
            text=response,
        )
    )
