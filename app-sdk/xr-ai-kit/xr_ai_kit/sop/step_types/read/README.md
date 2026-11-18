# Read step

`ReadStep` reads something from the camera into a schema, frame after frame,
until what it has read is enough: a tea's package, a label, a nameplate, a
form. Each frame's reply is turned into the record that is kept (`extract`),
and the records are merged field by field, so the front of a box and its back
add up, and a frame that does not show a field does not erase it. The result
is the merged record.

```python
identify = ReadStep(
    DEFINITIONS["identify"],          # start_line, done_line, and its `read: {question}` section
    perceive,                         # a vision model; thinking off (see below)
    reply_schema=TeaPackageLook,      # what the model fills in for one frame
    extract=tea_from_look,            # one reply -> the record kept (a Tea)
    is_enough=lambda tea: tea.tea_type is not None,
    settings=ReadSettings(interval_s=0.0, timeout_s=30.0),
)
tea = await run.step(identify)        # a Tea, or None when the person skipped it
```

`done_line` may name `{reading}` (the record): `I read {reading.spoken_name}.`

## The reply schema is a grammar

How to shape what the model fills in is in [the step types' README](../README.md#shaping-a-schema-for-a-vision-model):
gate on something plainly visible, gate each group of fields the model may
fill from memory, copy the words before interpreting them, and quote the
words each value comes from. The tea sample's (`tea.py`):

```text
look:
  | { tea_package_in_view: false, what_is_in_view }
  | { tea_package_in_view: true,
      text_on_package: [...],                         # the words, copied first
      identity: { brand:    {quote, value} | null,
                  name:     {quote, value} | null,
                  tea_type: {quote, value} | null },   # may be inferred: Premium Green Tea -> green
      directions:
        | { brewing_directions_visible: false }
        | { brewing_directions_visible: true, quote, water_temperature, steep_time } }
```

`extract` is where the reply's shape stops: it returns the plain record (no
gates, no quotes) the rest of the procedure uses, and an empty record for a
frame without the thing, which the merge ignores.

## What works and what does not

Measured on the green-tea recording, `perceive` with thinking off, 18 fixed
frames read three times each (2026-11-18):

| Reply schema | Frames without a package that produced a value | Label frames read right | Brewing settings invented on label frames |
|---|---|---|---|
| a source before every value, a value only from the package's text, no gate | 7 of 24 | 15 of 15 | 15 of 15 |
| gate + copied words + quotes | 0 of 24 | 14 to 15 of 15 | 6 to 8 of 15 |
| the same + a "where from" label on every field | 0 to 1 of 24 | 11 to 15 of 15 | 11 to 15 of 15 |
| **gate + copied words + quotes + a gate for the directions (the sample's)** | **0 of 24** | **14 to 15 of 15** | **0 to 1 of 15** |
| the sample's + a gate "is the name fully legible?" | 0 of 24 | 14 of 15 | 1 of 15 (no better) |
| copied words + quotes, no gate | 24 of 24 (a kind of tea inferred on every frame) | 14 of 15 | some |

The label's front prints no brewing directions, so every setting reported on
a label frame was invented (the model knows the brand's directions). As a step
(the kettle in view, then the label): the sample's schema reads Bigelow
Premium Green, and no brewing temperature, in 3 of 3 runs, in about 6 s. Plain
fields with the rules in the question took the kettle's display (68) for a
brewing temperature of 68 °C in 3 of 3, and the settings step kept it.

Still weak: a label lying sideways or half out of view is sometimes read as a
different tea ("yellow lemon tea" for the Bigelow bag), or with the kind of
tea written as the brand and the name ("green green"). The legibility gate did
not help: a model cannot judge the quality of its own reading. When it
matters, end the read on what the person shows clearly (`is_enough` asking for
a brand and a name, not only a kind), or confirm in the next step.

## Smaller models

- Keep the gates: each is a yes/no a small model can make, and a frame it is
  unsure of reads nothing instead of something wrong.
- Fewer fields per reply: read only what the procedure needs; leave what can
  be worked out to a transform step (the tea sample reads the identity and
  only printed directions; the settings are filled in afterwards).
- `extract` and the merge are code: a small model only has to get one frame
  right at a time, and any frame that does is enough.

## Speed

- Thinking off: about 1 s a frame for the omni model. With thinking on, the
  same frames took up to 11 s each.
- `interval_s: 0` (the default) reads the next new frame as soon as the last
  read is done; a frame is never read twice.
- `is_enough` ends the read at the first frame that completes it.
