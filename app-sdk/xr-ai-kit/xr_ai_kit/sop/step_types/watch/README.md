# Watch step

`WatchStep` confirms something with the camera: the kettle has water, the
display reads at least the target, the tea is in the water. It looks at a
frame every `interval_s`; a vision model describes it (`perceive`), a fast
classifier labels the description `positive`, `negative` or `unknown` against
the step's criteria (`decide`), and the step completes when `required` of the
last `window` decidable frames are positive. Unknown frames take no place in
the window, so an obscured view neither confirms nor resets it. It never
reasons and never asks; an escalator beside it can (`..escalation`).

```python
heat_water = StepWithEscalator(
    WatchStep(
        DEFINITIONS["heat_water"],                    # its `watch: {question, criteria}` section
        WatchModels(perceive=perceive, decide=decide),
        inputs_schema=HeatInputs,                     # {minimum} in its criteria
        facts_schema=DisplayFacts,                    # what the person may need to say (the unit)
        observation=StructuredObservation(            # read each frame into a schema
            schema=Thermometer, apply_known_facts=use_known_unit, can_judge=can_judge_reading
        ),
    ),
    ReasoningEscalator(DEFINITIONS["heat_water"], answer_with_reasoning),
)
await run.step(heat_water, HeatInputs(target=WaterTemperature(value=80, unit="celsius")))
```

## Patterns

- **Describe, then decide.** The perceiving model only says what it sees (free
  text, or a schema); the deciding model only labels that text against the
  criteria. Each can be read back and evaluated on its own, and the decision
  is a short classification a fast model makes well.
- **Criteria with an explicit unknown.** `unknown` says what does not count:
  nothing relevant in view, a picture of the thing (packaging art, a photo),
  the view obscured. Without it, a look-alike is labelled one way or the other.
- **Known facts win in code.** A schema observation (`StructuredObservation`)
  goes through `apply_known_facts` (a unit the person stated replaces one the
  model claims to read) and `can_judge` (no unit known or read: the frame is
  unknown, and nothing decides it) before the deciding model sees it. Asking
  the deciding model to prefer the known facts is less reliable than not
  showing it the wrong ones.
- **What the camera cannot show comes from the person.** `facts_schema` names
  it; the escalator beside the step asks when the step is stuck, and the
  answer is added to the step's context, which the next frame's decision reads.
- **Words that depend on the run are inputs.** `{minimum}` in the criteria is
  a property of the step's inputs, so the step is defined once.

## What works and what does not

The kettle display (heat_water's `Thermometer`), measured on the green-tea
recording, `perceive` with thinking off, 12 frames read three times each
(2026-11-18). The display never prints a unit; for 3 s at the start it shows
the target setting, 176, while a button is pressed:

| Observation schema | A unit invented | Setpoint frames read as current | 176 read right | 163 read right |
|---|---|---|---|---|
| plain fields (`value`, `unit_text`, `role`) | 4 of 36 | 8 of 9 | 6 of 6 | 0 of 3 ("16.3") |
| a gate (display in view?), the characters copied first, gates for "unit printed?" and "indicator shown?" | 3 of 36 | 3 of 9 | 6 of 6 | 2 of 3 |

- Copying the display's characters before the number helps the number.
- Neither schema makes the model reliable about units it is not shown, so the
  unit comes from code (`apply_known_facts`, from what the person said) and an
  unknown unit means the frame is not judged (`can_judge`).
- Asking the model whether a number is a setpoint does not work either way:
  the plain schema missed most setpoints, and the gated one called the real
  reading at the target a setpoint in 3 of 3 tries, which would never complete
  the step. The tea sample keeps the plain `role`. That leaves a gap: a 3 s
  flash gives two frames at a 2 s interval, enough for 2 of 4, so once the
  unit is known a setpoint flash can complete the step. In the sample the unit
  is not known yet when the flash happens (the person is asked later), which
  is why the evaluation case `setpoint_flash` no longer tests it on its own.

The general rules for shaping an observation schema are in [the step types'
README](../README.md#shaping-a-schema-for-a-vision-model).

## Smaller models

- The two-stage split is what lets the deciding model be small: it reads a
  sentence and picks one of three labels (the tea sample uses Clef).
- Keep the perceiving question narrow: what to look at, what to report, and
  what is not the thing (artwork, photos, other objects).
- A schema observation is easier for a small perceiving model than free text
  when the criteria compare numbers; `can_judge` keeps its unreadable replies
  from being decided at all.
- Raise `required` and `window` (3 of 5) when a small model's labels are
  noisier; unknowns still cost nothing.

## Speed

- `interval_s` (2 s by default) spaces the frames: the check needs distinct
  views over time, not every frame. Each frame costs one perceiving call
  (about 1 s for the omni model, thinking off) and one decision (a classifier).
- `required` of `window` sets how fast it can complete: 2 of 4 at 2 s is about
  4 s once the thing is in view.
- The step keeps its last 8 judgments (with their frames) for an escalator;
  nothing else grows with time.
