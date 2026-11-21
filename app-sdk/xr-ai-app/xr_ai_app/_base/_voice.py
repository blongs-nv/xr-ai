# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default voice agent the app builds from its config and the selected profile."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xr_ai_models import ModelsConfig, make_tts

from .profiles import Profile
from .topics import INTERRUPTED, USER_QUERY

if TYPE_CHECKING:
    from xr_ai_voice import VoiceAgent

    from .app import App


def models_for_voice(profile: Profile, speech_in: str = "speech_in") -> ModelsConfig:
    """Return the profile's models with the entry that fills ``speech_in`` also named ``stt``.

    ``VoiceAgent.from_models`` looks the batch STT up by the entry name ``stt``, while
    profiles fill ``speech_in`` through a ``roles`` map. The copy leaves the profile unchanged; with
    no ``speech_in`` role the models pass through, so a plain ``stt`` entry still works.
    """
    try:
        entry = profile.resolve(speech_in)
    except KeyError:
        return profile.models
    entries = dict(profile.models.entries)
    entries["stt"] = entries[entry]
    return ModelsConfig(entries)


def _speech(app: App) -> dict[str, Any]:
    """The ``Voice`` component's speech settings (``stt``, ``tts``, ``gate``, ``text_input``), or the defaults."""
    from ..voice import Voice  # noqa: PLC0415 - the public module imports this package

    for component in app._components:
        if isinstance(component, Voice):
            return dict(component._speech)
    return {"stt": None, "tts": None, "gate": None, "text_input": True}


def _gate_path(app: App):
    speech = _speech(app)
    return (
        app._config.config_dir / speech["gate"]
        if speech["gate"] is not None
        else app._config.voice.voice_gate_yaml or app._config.config_dir / "voice_gate.yaml"
    )


def build_voice_agent(app: App, profile: Profile) -> VoiceAgent | None:
    """Build the voice agent for ``app``: speech models, voice gate, hub transport and topics.

    Returns ``None`` when ``voice.enabled`` is false. Speech in and out are the ``Voice`` component's ``stt`` and
    ``tts`` roles when it names them, else the profile's ``speech_in`` and ``speech_out`` roles, else its entries named
    ``stt`` and ``tts`` (what the voice package itself reads, so an upstream ``models.json`` works unchanged). The voice
    gate is ``Voice(gate=...)``, else ``voice.voice_gate_yaml``, else ``voice_gate.yaml`` beside ``app.yaml``. That
    file decides whether speaker ASR is on; a ``speech_in`` entry that is the speaker service turns it on at its own
    endpoint (any other ``speech_in`` leaves the file in force). The agent never installs signal handlers (R7.3);
    ``App.stop`` ends it through ``request_stop``. ``app`` closes the TTS client, waits for it (and every bound role)
    before reporting ready, and reports busy/idle through the transport's endpoint.
    """
    from xr_ai_voice import HubVoiceTransport, VoiceAgent  # noqa: PLC0415
    from xr_ai_voice._processors import VadConfig  # noqa: PLC0415
    from xr_ai_voicegate import load_voice_gate_config  # noqa: PLC0415

    config = app._config
    if not config.voice.enabled:
        return None
    speech = _speech(app)
    speech_in = getattr(speech["stt"], "name", None) or "speech_in"
    speech_out = getattr(speech["tts"], "name", None) or ("speech_out" if _has(profile, "speech_out") else "tts")
    url = profile.speaker_asr_base_url(speech_in)
    gate_kwargs: dict[str, Any] = {} if url is None else {"speaker_base_url": url, "speaker_enabled": True}
    gate_file = _gate_path(app)
    voice_gate = load_voice_gate_config(gate_file, **gate_kwargs)
    entry = profile.resolve(speech_out)
    tts = make_tts(profile.models, entry)
    app._closeables.append(tts)
    app._probes.setdefault(entry, tts.health)
    transport = HubVoiceTransport()
    app._agent_status = transport.endpoint.set_status
    return VoiceAgent.from_models(
        models=models_for_voice(profile, speech_in),
        query_topic=USER_QUERY,
        tts=tts,
        vad=config.voice.vad or VadConfig(),
        voice_gate=voice_gate,
        ready_file=app._ready_file,
        text_input=bool(speech["text_input"]),
        participant_joined_topic=None,  # presence has one publisher, the Hub component (D-323a, R8.4)
        participant_left_topic=None,
        interrupted_topic=INTERRUPTED,
        interrupt_on_supersede=config.voice.interrupt_on_supersede,
        idle_timeout_secs=config.voice.idle_timeout_s,
        transport=transport,
        install_signal_handlers=False,
    )


def _has(profile: Profile, role: str) -> bool:
    try:
        profile.resolve(role)
    except KeyError:
        return False
    return True
