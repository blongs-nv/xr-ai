# Step types

| Package | Step | What it does | Models |
|---|---|---|---|
| [`watch/`](watch/README.md) | `WatchStep` | the camera confirms something: a frame at a time, perceive then decide, until N of the last M decidable frames are positive | a vision model, a fast classifier |
| [`read/`](read/README.md) | `ReadStep` | reads something from the camera into a schema, frame after frame, until it has enough | a vision model |
| [`transform/`](transform/README.md) | `TransformStep` | a reasoning model turns one schema into another, following the step's prompt; may ask the person | a reasoning text model |
| [`wait/`](wait/README.md) | `WaitStep` | a countdown | none |

Every step type has the same shape (`..steps.step_api`): it is defined once
from its id, its words (its type's words schema, loaded with `load_words`),
the models it uses and its `inputs_schema`, and each run gets its inputs:
`await run.step(step, inputs)`.
An app's own step type follows the same shape, and is evaluated the same way.

Each package's README covers what works, what does not, how it works with
smaller models, and how to make it faster. This one covers what they share:
how to shape a schema a vision model fills in.

## Shaping a schema for a vision model

A schema the model fills in for one frame is a grammar: with constrained
decoding (an endpoint that declares `json_schema`), the model can only write
what the schema allows, in the order it lists. The order and the choices are
the model's chain of thought when it does not reason (thinking off), and they
decide what it can get wrong.

Measured on the green-tea recording, with the omni model as `perceive`
(thinking off), 18 fixed frames read three times each (temperature 0, and
0.7 twice), each answer checked against the frame by eye (2026-11-18). The
numbers are from one recording; the direction of each effect was clear.

**What works**

1. **Gate on something plainly visible, first.** A top-level choice between
   two shapes, `{package_in_view: false, what_is_in_view}` or
   `{package_in_view: true, ...the fields}`, so a frame without the thing
   cannot produce any value. Frames without a tea package that produced a
   value: 7 of 24 without a gate, 0 of 24 with it.
2. **Gate each group of fields the model is tempted to fill from memory.**
   `{directions_visible: false}` or `{directions_visible: true, quote,
   temperature, time}`. Brewing settings invented on frames whose package
   prints none (the model knows the brand's directions): 6 to 8 of 15 without
   the gate, 0 to 1 of 15 with it.
3. **Copy before you interpret.** A list of the words on the thing, or the
   characters on a display exactly as shown, before the fields that interpret
   them. A display showing 163 was read as "16.3" in 3 of 3 tries from a
   number field alone, and as 163 in 2 of 3 once the characters were copied
   first.
4. **Quote the evidence with each value.** `{quote: "Premium Green Tea",
   value: "green"}`: the value is tied to words in view, and a value that may
   be inferred (the kind of tea) says what it is inferred from.
5. **Name the object before reading it.** "A lit display on a kettle", or
   "only printed text or a label": a tea bag's "100% Organic" was read as a
   display's temperature in 11 of 20 tries until the caption had to say what
   it was reading, then in none.
6. **Constrain, do not explain.** Closed choices over free text (`Literal`
   enums for kinds, units, roles; a pattern for digits, `^[0-9]{1,3}$`, which
   stopped "163." becoming 16.3), and units kept as printed, with conversion in
   code. Explaining instead backfired: a description saying "the dot is a
   degree mark" made the model invent units in 16 of 60 readings.

**What does not work**

- **Gating on a judgement the model cannot make about itself.** "Is the name
  fully legible?" did not stop a sideways label being read as a different
  tea; "does an indicator mark this as a setpoint?" turned real readings
  into setpoints (3 of 3 at the target, which would never complete the
  check). A gate works when its question is about the scene (is X in view,
  is text of a kind printed), not about the quality of the model's own
  reading.
- **A "where from" label on every field.** `{source: own_knowledge |
  other_object_text | package_text, value}` per field made invented settings
  more frequent (11 to 15 of 15): the model labelled what it knows about a
  brand as printed, and listing the alternatives seemed to prompt it to
  think of the settings. It decides well about a whole object, poorly per
  field.
- **A plain schema with the rules in the question.** "Only what is printed on
  the package" in the question did not stop the kettle's display (68) being
  read as a brewing temperature of 68 °C.
- **Small changes deciding the outcome.** Without the gates, making one
  field plain instead of sourced brought the kettle's 68 back in every try.
  Measure a schema change; do not reason about it.
- **Class docstrings on a schema.** They become the schema's description, and
  with one the reasoning model's constrained reply left a field out in 6 of 6
  tries. Put the meaning in field descriptions and comments.

**Thinking instead of a grammar.** With thinking on, the same model did not
take the kettle's display for a brewing temperature either, but a frame of the
label took about 11 s instead of about 1 s, and one read ran past the end of a
12 s scene. A grammar costs nothing per frame; use thinking where a step is not
in a per-frame loop (transform, escalation).

**How to measure a schema.** Pick frames for each situation (the thing in
view, the thing absent, look-alikes, the thing partly in view), look at each
to know what is true, call the model directly with each schema variant (a few
times, at temperature 0 and above), and count separately what it got right and
what it invented: a schema that reads more can also invent more.

## Smaller models

- Every gate is a binary choice a small model can make; every free-text field
  is a place to go wrong. Prefer enums, gates and short copied strings.
- Keep unions shallow: one gate at the top, one per group of fields.
- Constrained decoding matters more for a small model: without
  `json_schema` on the endpoint the schema is only described in the prompt,
  and replies that do not parse are lost frames.
- Keep `max_tokens` small for per-frame calls (the tea sample: 512), the
  temperature at 0, and the question short; the shape carries the
  instructions.
- Expect more unreadable replies: the read and watch steps treat one as a
  frame without a decision, and keep going.

## Speed

- Per-frame models run without thinking (about 1 s a frame for the omni model
  here); reasoning belongs off the per-frame loop.
- `ReadStep` reads the next new frame as soon as the last is read
  (`interval_s: 0`); a watch step looks every `interval_s` (2 s by default),
  since it needs distinct frames over time, not as many as possible.
- Stop early: `is_enough` ends a read as soon as what was read is enough, and
  a watch step completes on N of the last M decidable frames.
