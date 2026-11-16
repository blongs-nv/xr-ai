# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Component``: the base of everything added to an app (``app.add(...)``)."""

from __future__ import annotations

from types import FunctionType
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Coroutine

    from .app import App
    from .participant import Participant

T = TypeVar("T")

__all__ = ["Component"]


class Component:
    """Base class of app components: lifecycle hooks, ``self.app`` and ``self.spawn``.

    Hooks are optional and default to no-ops. They run in a fixed order:
    ``on_start`` and ``on_participant_enrolled`` in registration order, ``on_participant_unenrolled`` and
    ``on_stop`` in reverse registration order (see ``LIFECYCLE_CONTRACT.md``). Component code never creates tasks
    (use ``p.spawn``, or ``self.spawn`` for work that belongs to no participant), never catches ``CancelledError``
    and never orders shutdown. A component is routable through an ``xr_ai_kit.RouterDestination`` attribute it builds
    from its own methods. A component is not a runtime agent: a class with a foundation ``@subscribe`` handler raises
    ``TypeError`` when it is defined.
    """

    _app: App | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject a foundation ``@subscribe`` handler: a component is not a runtime agent, so it would never run."""
        super().__init_subclass__(**kwargs)
        for klass in cls.__mro__:
            for name, value in vars(klass).items():
                handler = getattr(value, "__func__", value)  # staticmethod and classmethod wrap the function
                if isinstance(handler, FunctionType) and getattr(handler, "__xr_ai_topics__", None):
                    raise TypeError(
                        f"{cls.__name__}.{name} is a foundation @subscribe handler, but a Component is not a runtime "
                        "agent and is never subscribed to topics. Use the component hooks (on_start, "
                        "on_participant_enrolled, on_participant_unenrolled, on_stop), app.on_turn for user turns, "
                        "and p.connected for a participant's connection."
                    )

    @property
    def app(self) -> App:
        """The app this component is added to; unavailable before ``app.add(component)``."""
        app = self._app
        if app is None:
            raise RuntimeError("component is not added to an app; call app.add(component) first")
        return app

    async def on_start(self) -> None:
        """Called once after models are ready and before any participant is admitted."""

    async def on_participant_enrolled(self, p: Participant) -> None:
        """Called when a participant is enrolled, after its scope exists."""

    async def on_participant_unenrolled(self, p: Participant) -> None:
        """Called after the un-enrolled participant's tasks drained, before its resources are released.

        Runs in reverse registration order.
        """

    def spawn(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        key: str | None = None,
        supersede: bool = False,
    ) -> asyncio.Task[T]:
        """Run ``coro`` for the component's lifetime; the app owns the task and cancels it at stop.

        Unlike :meth:`Participant.spawn` the work belongs to no participant: it survives every un-enrollment.
        ``key`` and ``supersede`` mean what they do on ``Participant.spawn``, in a namespace of this component only.
        Work started in ``on_start`` runs at once; it does not wait for the app to admit participants.
        Raises ``RuntimeError`` when the component is not added to an app or the app is stopping, and
        ``ValueError`` when ``key`` is live and ``supersede`` is false.
        """
        try:
            app = self.app
        except RuntimeError:
            coro.close()
            raise
        return app._spawn_component_task(self, coro, key=key, supersede=supersede)

    async def on_stop(self) -> None:
        """Called once after every participant has left and before model clients close."""
