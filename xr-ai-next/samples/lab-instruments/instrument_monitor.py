# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The instrument monitor: read the instruments every few seconds, speak up
when one is found, when its reading changes, and when it is lost.

One background task per participant (`p.spawn`), started and stopped by the
conversation's tools. Readings are compared by number and unit
(`normalize_meter_reading`, upstream's), so "23.50 C" after "23.5 C" is no
change. Every change and loss is said, written to `instrument-monitoring.jsonl`
and added to the monitoring history the conversation reads; a snapshot of all
tracked instruments is written every `state_every_s`.
"""

import asyncio
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from instruments import InstrumentReader, Reading, Sighting
from records import (
    InstrumentChange,
    InstrumentLost,
    InstrumentState,
    InstrumentStateSnapshot,
    Records,
)
from xr_ai_next import Component, ContextLog, Participant, Sink, clock, trace

KEY = "instrument-monitor"
FILE = "instrument-monitoring"


@dataclass
class Tracked:
    value: Decimal
    unit: str
    state: InstrumentState
    last_seen: float  # clock.now()


class InstrumentMonitor(Component):
    def __init__(
        self,
        reader: InstrumentReader,
        *,
        output: Sink,
        records: Records,
        history: ContextLog,
        every_s: float,
        state_every_s: float,
        lost_after_s: float,
    ) -> None:
        super().__init__()
        self._reader = reader
        self._output = output
        self._records = records
        self._history = history
        self._every_s = every_s
        self._state_every_s = state_every_s
        self._lost_after_s = lost_after_s

    def running(self, p: Participant) -> bool:
        task = p.state.get(KEY)
        return task is not None and not task.done()

    def start(self, p: Participant) -> str:
        if self.running(p):
            return "Lab instrument monitoring is already running."
        p.state[KEY] = p.spawn(self._watch(p), key=KEY, detached=True)
        return "Lab instrument monitoring started."

    async def stop(self, p: Participant) -> str:
        if not await p.cancel(KEY):
            return "Lab instrument monitoring is not running."
        return "Lab instrument monitoring stopped."

    def status(self, p: Participant) -> str:
        if self.running(p):
            return "Lab instrument monitoring is running."
        return "Lab instrument monitoring is stopped."

    async def _watch(self, p: Participant) -> None:
        tracked: dict[tuple[str, str], Tracked] = {}
        next_state = clock.now() + self._state_every_s
        while True:
            try:
                result = await self._reader.read(p)
            except Exception as exc:  # noqa: BLE001 - try again next time
                trace.emit("error", KEY, error=type(exc).__name__)
            else:
                now = clock.now()
                for event in observe(
                    tracked, result.readings, result.sightings, now
                ):
                    await self._report(p, event)
            for event in lost(tracked, clock.now(), self._lost_after_s):
                await self._report(p, event)
            if clock.now() >= next_state:
                next_state = clock.now() + self._state_every_s
                states = sorted(
                    (t.state for t in tracked.values()),
                    key=lambda s: s.device_name,
                )
                snapshot = InstrumentStateSnapshot(
                    timestamp_us=clock.wall_us(), instruments=list(states)
                )
                await self._records.write(p, FILE, snapshot)
            await asyncio.sleep(self._every_s)

    async def _report(
        self, p: Participant, event: InstrumentChange | InstrumentLost
    ) -> None:
        text = spoken(event)
        await p.say(text, to=self._output)
        await self._history.publish(p, text)
        await self._records.write(p, FILE, event)


def spoken(event: InstrumentChange | InstrumentLost) -> str:
    """What the monitor says (upstream's InstrumentAlertAgent)."""
    name, reading = event.device_name, event.meter_reading
    if isinstance(event, InstrumentLost):
        return (
            f"I am no longer tracking {name}. Its last reading was {reading}."
        )
    if event.change_type == "discovered":
        return f"Now tracking {name} at {reading}."
    return f"{name} changed from {event.previous_reading} to {reading}."


def observe(
    tracked: dict[tuple[str, str], Tracked],
    readings: list[Reading],
    sightings: list[Sighting],
    now: float,
) -> list[InstrumentChange]:
    """Update `tracked` from one read; the discoveries and changes."""
    for seen in sightings:  # a marker in view keeps its instrument tracked
        if (known := tracked.get((seen.marker_type, seen.marker_id))) is None:
            continue
        known.last_seen = now
        known.state.last_seen_us = seen.timestamp_us
        known.state.tracking = True
    changes = []
    for reading in {(r.marker_type, r.marker_id): r for r in readings}.values():
        key = (reading.marker_type, reading.marker_id)
        known = tracked.get(key)
        normalized = normalize_meter_reading(
            reading.meter_reading, previous_unit=known.unit if known else ""
        )
        if normalized is None:  # no number in it
            continue
        value, unit, display = normalized
        change = InstrumentChange(
            timestamp_us=reading.timestamp_us,
            change_type="discovered" if known is None else "reading_changed",
            marker_type=reading.marker_type,
            marker_id=reading.marker_id,
            device_name=reading.device_name,
            previous_reading=None
            if known is None
            else known.state.meter_reading,
            meter_reading=display,
            last_seen_us=reading.timestamp_us,
        )
        if known is None:
            state = InstrumentState(
                marker_type=reading.marker_type,
                marker_id=reading.marker_id,
                device_name=reading.device_name,
                meter_reading=display,
                first_seen_us=reading.timestamp_us,
                last_seen_us=reading.timestamp_us,
            )
            tracked[key] = Tracked(value, unit, state, now)
            changes.append(change)
            continue
        known.last_seen = now
        known.state.last_seen_us = reading.timestamp_us
        known.state.tracking = True
        known.state.meter_reading = display
        if (value, unit) != (known.value, known.unit):
            known.value, known.unit = value, unit
            changes.append(change)
    return changes


def lost(
    tracked: dict[tuple[str, str], Tracked], now: float, after_s: float
) -> list[InstrumentLost]:
    """The instruments not seen for `after_s`, once each until seen again."""
    events = []
    for known in tracked.values():
        if not known.state.tracking or now - known.last_seen < after_s:
            continue
        known.state.tracking = False
        events.append(
            InstrumentLost(
                timestamp_us=clock.wall_us(),
                marker_type=known.state.marker_type,
                marker_id=known.state.marker_id,
                device_name=known.state.device_name,
                meter_reading=known.state.meter_reading,
                last_seen_us=known.state.last_seen_us,
            )
        )
    return events


# -- comparing readings (upstream's normalize_meter_reading) -----------------

_NUMBER = re.compile(r"[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][+-]?\d+)?")
_UNIT = re.compile(r"(?:°\s*)?[A-Za-zµμΩ%]+(?:\s*/\s*[A-Za-z]+)?")
_UNIT_ALIASES = {
    "amp": "A",
    "amps": "A",
    "ampere": "A",
    "amperes": "A",
    "celsius": "°C",
    "degree c": "°C",
    "degrees c": "°C",
    "fahrenheit": "°F",
    "degree f": "°F",
    "degrees f": "°F",
    "hertz": "Hz",
    "ohm": "Ω",
    "ohms": "Ω",
    "percent": "%",
    "volt": "V",
    "volts": "V",
    "watt": "W",
    "watts": "W",
}


def normalize_meter_reading(
    reading: str, *, previous_unit: str = ""
) -> tuple[Decimal, str, str] | None:
    """(number, unit, display) of a reading; the unit is kept from before
    when the display shows none. None when it holds no number."""
    match = _NUMBER.search(reading)
    if match is None:
        return None
    token = match.group(0)
    token = (
        token.replace(",", "")
        if "," in token and "." in token
        else token.replace(",", ".")
    )
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    unit = previous_unit
    suffix = reading[match.end() :].strip(" :=-_")
    if (unit_match := _UNIT.match(suffix)) is not None:
        raw = re.sub(r"\s+", " ", unit_match.group(0)).strip()
        unit = _UNIT_ALIASES.get(raw.casefold(), raw.replace(" ", ""))
    number = format(value, "f")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return value, unit, f"{number} {unit}".strip()
