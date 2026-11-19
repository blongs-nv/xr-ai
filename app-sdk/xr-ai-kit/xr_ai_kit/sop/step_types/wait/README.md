# Wait step

`WaitStep` counts down, reporting the time left every `interval_s`; no
camera, no models. Its duration is either fixed, or each run's input when it
is known only once earlier steps have run:

```python
warm_up = WaitStep("warm_up", load_words(STEPS / "warm_up" / "words.yaml", StepWords), duration_s=90)
# a fixed duration's words may name {duration_s} and {minutes}: "Warming up for {duration_s} seconds."


class SteepInputs(WaitInputs):        # duration_s, and {minutes} for its words
    @property
    def minutes(self) -> float:
        return self.duration_s / 60


steep = WaitStep("steep", load_words(STEPS / "steep" / "words.yaml", StepWords), inputs_schema=SteepInputs)
await run.step(steep, SteepInputs(duration_s=150))
```

- Time is read through `xr_ai_app.clock`, so the devkit's virtual clock runs a
  three-minute countdown in moments in tests and evaluations.
- The last sleep is bounded by the time left, so a long `interval_s` never
  delays completion.
- Skipping it ends the countdown early; the procedure continues.

Nothing here depends on a model, so there is nothing to tune for smaller
ones; `interval_s` only sets how often the time left is reported (for the
conversation's "how long is left?").
