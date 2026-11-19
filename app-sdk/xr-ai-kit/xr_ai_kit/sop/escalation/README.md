# Escalation

An escalator runs beside one step (`StepWithEscalator`) and helps the person
when the step cannot decide. The kit's, `ReasoningEscalator`, looks at the
step's recent evidence every `every_s`; when most of it is undecided
(`is_stuck`), a reasoning model reads the step's task, `guidance`, the
evidence and what is known, and replies with one decision (simplified: the
real schema gives each field a description):

```python
class _Decision(BaseModel):
    missing: str                                # what the check is missing
    action: Literal["none", "ask", "tell"]      # ask: wait for an answer; tell: say it
    words: str | None                           # the spoken line
```

An `ask` waits for the answer and passes it to the step as context, read into
the step's `facts_schema`; a `tell` is only said. Choosing `tell` for a line
that needs an answer ("State the unit shown") loses the answer: the step
never learns it.

## What the person says without being asked

With `learns_from_conversation`, the escalator also reads the conversation
(`environment.conversation`) into the step's `facts_schema`: once as the
step starts, and again each time the conversation changes, stuck or not.
The unit said when the target was announced, or volunteered while the water
heats, reaches the step without anyone asking:

```
the step starts ───────────────┐
the conversation changes ──────┤  (environment.conversation.changes(): no polling)
                               └─ read the conversation into facts_schema
                                    └─ anything new? → add it to the step's context
                                         → the judge has it on the next frame
```

- A person can assert only what `facts_schema` names: "it's in Fahrenheit",
  never "the water is hot" (that is the camera's to judge). A later
  statement replaces an earlier one.
- The escalator says nothing while the conversation is being read, or right
  after it gave the step something new: the step decides with that first.
- Nothing is spoken: the conversation answers the person's turn as usual.

Its three switches, in `EscalationSettings`:

| Setting | What it does |
|---|---|
| `learns_from_conversation` | the conversation is read into the step's facts (above) |
| `reads_conversation` | the conversation goes into the escalator's own prompt when it decides what to say |
| `reads_run_summary` | the run's summary (the plan, the steps' states) goes into that prompt too |

The reader, measured on 14 tea dialogues (2026-11-19), every one with the
assistant saying "Heat the water to 80 degrees Celsius": 6 where the person
states no unit and says nothing about one that could pass for it ("okay",
"what's that in Fahrenheit?", "what if it were Fahrenheit?", a question left
unanswered), 2 where they hedge or talk of another kettle, and 6 where they
state it (volunteered, answered, corrected).

| Reader | Right | A unit from the assistant's words or a question (of the 6) | A unit from a hedge | Missed a stated unit |
|---|---|---|---|---|
| **fill in only what the person stated, read twice, kept only where both agree (the kit's)** | **131 of 140 pairs** | **0 of 60** | 8 of 20 | 1 of 60 |
| the same, read once | 128 of 140 | 2 of 90 (and once in a real-model run, which completed the step at 82 °F) | 11 of 20 | 0 |
| the same, "about what they are using now" | 53 of 60 | 5 of 25 | 2 of 10 | 0 |
| only the person's lines, without the assistant's | 120 of 140 | 8 of 60 ("what's that in Fahrenheit?" read as Fahrenheit) | 11 of 20 | 1 |
| a gate first: "has the person stated it?", their words copied | 62 of 70 | 0 of 30 | 8 of 10 | 0 |
| the same, their line copied (pattern `^user: `) | 55 of 70 | 0 of 30 | 10 of 10 | 5 |

- Read once, a stray fill is rare, but the history is read on every change
  and a fact once given stays: over a step, the rare becomes likely. Two
  reads that must agree remove the stray fills and cost no time (they run
  at once).
- A hedge ("I usually think in Celsius, not sure what this kettle does")
  is still read as a statement, by every reader: agreement does not catch
  a systematic misreading. It is the hole that remains.
- Rewording the instruction made it worse; so did hiding the assistant's
  lines (the reader needs the answer to "what's that in Fahrenheit?" to
  know it was a question about a conversion).
- A wrong fact makes the check judge wrongly (a unit read as Celsius
  completes a Fahrenheit kettle at 82), so a step whose facts are risky to
  get wrong can leave `learns_from_conversation` off and rely on asking.

## What works and what does not, deciding what to say

Measured on the green-tea recording (heat_water, 2026-11-19): 25 snapshots
the escalator was handed when stuck, captured from real-model runs (the
display seen with no unit known; the display out of view, with and without
the unit known), each replayed with the reasoning model (thinking on) and
each decision schema 3 to 15 times. Right means the line goes where it
should: a question about the unit when the display is seen and the unit is
not known, an instruction to show the display when it is out of view.

| Decision schema | Right | Mean time |
|---|---|---|
| **`action` ask/tell/none, then free words (the kit's)** | **347 of 375** | 8 s |
| the same, actions described by where the reply goes ("the person replies in words" / "does something the camera will see") | 214 of 225 | 8 s |
| one branch per action, the line's form constrained (a question ends in "?") | 68 of 75 | 11 s |
| the line first, then "does it expect a spoken answer?" | 69 of 75 (18 lines badly formed) | 10 s |
| a gate first: "is the item seen in the recent reports?", with a quote | 60 of 75 (6 unreadable) | 32 s |
| that gate with the line's form constrained | 54 of 75 (13 unreadable) | 39 s |
| "where will the information come from: the person's answer / the camera / nowhere" | 65 of 75 (43 lines badly formed: "State the unit", "Celsius") | 10 s |
| that, after copying what is already known | 55 of 75 | 14 s |
| "what is missing: a fact the person knows / something they must do / nothing" | 17 of 75 | 12 s |

- The kit's schema is as good as any measured; the alternatives are within
  noise of it or worse. Same-run comparisons with the best alternative were
  70 to 73 and 140 to 141.
- A request for information phrased as an instruction ("Tell me the unit",
  "Specify whether...") is sometimes filed as `tell`, about 1 in 40. A
  schema that asks the model to classify the missing thing as a fact or an
  action makes this the rule (58 of 75 wrong): to the model, "tell me the unit" is
  something the person must do.
- Gates that help a perceiving model (see
  [the step types' README](../step_types/README.md#shaping-a-schema-for-a-vision-model))
  did not help here. The reasoning model already reasons over the reports;
  a gate adds a judgement it can get wrong, slows it down (32 to 39 s) and
  makes replies unreadable.
- Forcing a question to end in "?" makes it a question in form only, and
  produced unreadable replies.
- The weakest situation is content, not form: with the unit already known
  and the display out of view, the reasoning model sometimes asks for the
  unit again, or to "show the unit label", instead of asking for the display
  to be brought back into view.

## Smaller models

- Keep the decision small: one enum and one line. Each extra gate or quoted
  field is a place for a smaller model to go wrong, and costs time.
- Thinking stays on: the escalator is off the per-frame loop, every
  `every_s` at most.

## Speed

- About 8 s a decision with thinking on; it runs only while the step is
  stuck, at most `max_lines` times, `min_gap_s` apart.
