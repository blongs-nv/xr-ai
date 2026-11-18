# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The kit's step types, one package each; an app's own step types follow the same shape (see ``..steps.step_api``).

=============  ===========================================================================================
``watch``      :class:`~.watch.WatchStep`: the camera confirms something; an optional per-step escalator
               helps when it cannot
``read``       :class:`~.read.ReadStep`: read something from the camera into a schema until it is enough
``transform``  :class:`~.transform.TransformStep`: a reasoning model turns one schema into another (it may ask the
               person); not skippable
``wait``       :class:`~.wait.WaitStep`: a countdown
=============  ===========================================================================================

Each package has the step, what only it uses, and its ``*_scoring`` module for evaluation, which the running code
never imports. They depend on ``..steps`` and never on each other.

Each package's README.md covers its patterns, what works and what does not, smaller models and speed; this
package's README.md covers shaping a schema a vision model fills in.
"""
