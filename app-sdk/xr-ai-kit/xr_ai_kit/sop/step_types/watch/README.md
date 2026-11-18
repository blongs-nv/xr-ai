# Watch step

`WatchStep` confirms something with the camera: the kettle has water, the
display reads at least the target, the tea is in the water. It looks at a
frame every `interval_s`; a vision model describes it (`perceive`), and the
description is judged `positive`, `negative` or `unknown`: by a fast
classifier against the step's criteria (`decide`), or by the app's code when
the description is a schema and the check is a comparison. The step completes
when `required` of the last `window` decidable frames are positive. Unknown
frames take no place in the window, so an obscured view neither confirms nor
resets it. It never reasons and never asks; an escalator beside it can
(`..escalation`).

```python
heat_water = StepWithEscalator(
    WatchStep(
        "heat_water",
        load_words(STEPS / "heat_water" / "words.yaml", WatchWords),   # question, criteria, lines
        WatchModels(perceive=perceive, decide=decide),
        inputs_schema=HeatWaterInputs,                  # the target; {minimum} in its criteria
        facts_schema=DisplayFacts,                      # what the person may need to say (the unit)
        observation=StructuredObservation(schema=DisplayCaption, judge=reached_target),
    ),
    ReasoningEscalator(load_words(STEPS / "heat_water" / "escalation.yaml", EscalationWords), reasoning),
)
await run.step(heat_water, HeatWaterInputs(target=WaterTemperature(value=80, unit="celsius")))
```

## Patterns

- **Describe, then judge.** The perceiving model only says what it sees; the
  judgement is separate. Each can be read back and evaluated on its own.
- **A structured caption when the frame has things to read.** A schema the
  perceiving model fills in (`StructuredObservation(schema=...)`), shaped as a
  grammar: see [the step types' README](../README.md#shaping-a-schema-for-a-vision-model)
  and the measurements below.
- **Judge in code what is a comparison; by the classifier what is a
  judgement.** "Is the reading at least 176 °F" is code (`judge=`): exact, and
  it can say `unknown` when the unit is missing. "Is there water in the
  kettle" stays with the classifier and the criteria.
- **Criteria with an explicit unknown.** `unknown` says what does not count:
  nothing relevant in view, a picture of the thing, the view obscured.
- **What the camera cannot show comes from the person.** `facts_schema` names
  it; the escalator asks when the step is stuck (most recent frames unknown),
  and the answer reaches the judge as the known facts. A judge that answers
  `unknown` without them is what makes the step stuck, and so makes the
  escalator ask.
- **Words that depend on the run are inputs.** `{minimum}` in the criteria is
  a property of the step's inputs, so the step is defined once.

## What works and what does not

The kettle display (heat_water), measured on the green-tea recording,
`perceive` with thinking off, each frame read five times (2026-11-18). The
display prints no unit; the target setting flashes for 3 s at the start; a tea
bag reading "100% Organic" sits in some frames with the display cut off.

| Caption | A unit invented | Number right | Digits read from a frame with no readable display |
|---|---|---|---|
| plain fields (`value`, `unit_text`, `role`) | 5 to 7 of 60 | 42 to 45 of 50 (163 read as "16.3") | 0 of 10 |
| free-text fields with evidence | 24 of 60 | 38 of 50 | — |
| + a gate: "is a unit letter printed?" | 0 of 60 | 41 to 43 of 50 | — |
| + the number as digits only (`^[0-9]{1,3}$`) | 1 of 60 | 46 of 50 | 11 of 20 |
| **+ what the display is on, what kind, where in the image (the sample's)** | **1 of 75** | **55 of 55** | **0 of 20** |
| + "the whole display is inside the image" | 3 of 75 | 55 of 55 | 4 of 20 |

- A gate for the unit stops invented units: the model writes C or F only
  after deciding a letter is printed.
- Digits only (a pattern) stops "131." and "163." becoming 13.1 and 16.3; an
  integer field with "the dot is a degree mark" in its description made the
  model invent units instead (16 of 60): constrain, do not explain.
- Naming the object first ("a lit display on a kettle", or "only printed
  text or a label") stopped the bag's "100%" being read as a temperature.
- A gate on the model's judgement of its own view ("the whole display is
  inside the image") made things worse.

Judging those descriptions (39 judgements per row):

| Judge | Wrong "done" while no unit is known | Wrong "done" once the unit is known | Target reached, found |
|---|---|---|---|
| the classifier, criteria in words | 9 to 22 | 8 to 9 | 7 to 8 of 9 |
| **code: the digits, the unit from the facts or a printed letter (the sample's)** | **1** | 9 | **9 of 9** |

The classifier decides "done" without a unit even when the caption says none
is printed and the criteria say that is unknown; code does not. Nothing tells
the target setting's 3 s flash from a real reading (all of the wrong "done"
once the unit is known): a flash gives two frames at a 2 s interval, enough for
2 of 4, so once the unit is known it can complete the step. In the sample the
unit is not known yet when the flash happens (the person is asked later).

## Smaller models

- Every gate and enum is a choice a small model makes well; a free-text field
  is where it invents. The caption's shape carries the instructions.
- Judging in code removes the second model from comparisons entirely.
- Keep the classifier's question short and its criteria concrete when it
  judges (the water, the tea bag); raise `required` and `window` (3 of 5) when
  its labels are noisier.

## Speed

- `interval_s` (2 s by default) spaces the frames: the check needs distinct
  views over time. Each frame costs one perceiving call (about 1 s for the omni
  model, thinking off; a structured caption is a little longer) and one
  classifier call, or none with a code judge.
- `required` of `window` sets how fast it can complete: 2 of 4 at 2 s is about
  4 s once the thing is in view.
- The step keeps its last 8 judgments (with their frames) for an escalator.
