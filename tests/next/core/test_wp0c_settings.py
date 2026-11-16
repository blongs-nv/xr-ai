# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``settings:`` section of app.yaml and ``App.settings``."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from xr_ai_next._base.app import App
from xr_ai_next._base.enrollment import PerConnection


class Tea(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int = 3
    brand: str


def _app(tmp_path: Path, body: str) -> App:
    path = tmp_path / "app.yaml"
    path.write_text(body)
    return App.from_config(path, enrollment=PerConnection(admit=None))


def test_settings_are_validated_with_a_pydantic_model(tmp_path: Path) -> None:
    app = _app(tmp_path, "settings:\n  brand: green\n")
    assert app._config.settings == {"brand": "green"}
    assert app.settings(Tea) == Tea(steps=3, brand="green")


def test_settings_default_to_empty_and_missing_required_fields_are_listed(tmp_path: Path) -> None:
    app = _app(tmp_path, "{}")
    assert dict(app._config.settings) == {}
    with pytest.raises(ValueError, match=r"invalid settings for Tea: brand: Field required"):
        app.settings(Tea)


def test_every_error_is_listed_and_other_unknown_top_level_keys_are_still_rejected(tmp_path: Path) -> None:
    app = _app(tmp_path, "settings: {steps: many, extra: 1}\n")
    with pytest.raises(ValueError) as err:
        app.settings(Tea)
    assert all(word in str(err.value) for word in ("steps", "extra", "brand"))
    (tmp_path / "bad.yaml").write_text("setting: {}\n")
    with pytest.raises(ValueError, match="unknown keys"):
        App.from_config(tmp_path / "bad.yaml", enrollment=PerConnection(admit=None))
    (tmp_path / "list.yaml").write_text("settings: [1]\n")
    with pytest.raises(ValueError, match="settings must be a mapping"):
        App.from_config(tmp_path / "list.yaml", enrollment=PerConnection(admit=None))
