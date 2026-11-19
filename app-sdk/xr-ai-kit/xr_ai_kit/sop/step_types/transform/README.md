# Transform step

`TransformStep` has a reasoning model turn one schema's instance into
another's, following the step's prompt (its words' `instructions`). No camera. Its inputs are what earlier steps produced; its result is
an instance of `output_schema`. Uses: fill in what a reading left empty, work
out settings from what is known, turn a description into a category.

```python
choose_settings = TransformStep(
    "choose_settings",
    load_words(STEPS / "choose_settings" / "words.yaml", TransformWords),   # instructions, lines
    answer_with_reasoning,             # a reasoning text model
    inputs_schema=Tea,
    output_schema=Tea,                 # the same schema: fill in the gaps
    keep_input_values=True,            # what was read is kept, in code
    required_fields=("water_temperature", "steep_time"),
)
tea = await run.step(choose_settings, read or Tea())   # nothing read: everything is filled in
```

- **`keep_input_values`**: every field the input has keeps its value; the
  model is told which are known and only fills the rest. It cannot overwrite
  what was read, whatever it replies.
- **`required_fields`** must be set in the result: a reply that leaves one
  empty is asked again (the prompt says which are missing), up to `attempts`
  replies, then the step fails, and with it the run. There are no silent
  defaults.
- **Asking the person**: when its instructions allow it, the model may reply
  with one short question instead of the output (`max_questions` per run, 1
  by default); the step asks it, and the answer goes into the next prompt.
  The question is the step's own (its recorded lines say `by: step`).
- **Not skippable**: it does not wait on the person to do anything, and later
  steps need its result, so the runner declines "skip" while it runs.

`done_line` may name `{output}`: `I'll brew it at {output.water_temperature.spoken}.`

## What works and what does not

The tea sample's `choose_settings` on the reasoning omni model
(`answer_with_reasoning`, thinking on), 2026-11-18:

| Input | Result |
|---|---|
| Bigelow, Premium Green Tea, green | 75 °C for 2 to 3 minutes (filled in) |
| green, 175 °F and 2 to 3 minutes printed | 175 °F and 2 to 3 minutes kept |
| nothing read; the person answers "It's a black tea." | asked "What kind of tea is this?", then 95 °C for 3 to 5 minutes |
| nothing read; no answer | asked, then 95 °C for 3 to 5 minutes |

- **Say what to do with little to go on.** "When nothing identifies the kind
  of tea, ask the person; if they cannot say, give settings that suit most
  teas" made it ask, and then commit to an answer. What "suits most teas"
  meant to it (95 °C, black-tea settings) is harsh for a green tea: name the
  fallback in the prompt when it matters.
- **Required fields plus retries** turn "the model left it out" into a second
  try with the missing field named, instead of a default nobody chose.
- **No class docstrings on the schemas**: a docstring becomes the schema's
  description, and with one the model left the temperature out of its reply
  in 6 of 6 tries.
- **Units as values with their unit** (`{value: 175, unit: fahrenheit}`), not
  converted by the model: a printed 175 °F stays 175 °F, and conversion is code.

## Smaller models

- Keep thinking on if the model has it: this step runs once, not per frame.
- Keep the output schema small and flat, with enums where the answer is one
  of a few; required fields and retries catch a small model's omissions.
- Put the fallback in the prompt (the value to use when unsure) rather than
  relying on what the model considers conventional.

## Speed

One reply, or two with a question, on the reasoning model: a few seconds
(about 4 to 9 s here), once per run. It is not in a per-frame loop, so
thinking is affordable.
