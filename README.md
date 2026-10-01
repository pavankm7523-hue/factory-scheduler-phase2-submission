# Factory Scheduler — Phase 2 Submission

A deadline-aware job-shop scheduling solver for the Factory Scheduler
challenge. The task is to decide the operation order on every shared machine
while respecting each job's operation sequence, release times, setup changes,
and weighted due dates.

> **Local public benchmark:** 45.714 / 100
>
> This is a local practice result from the supplied public suite, not an
> official score or a guarantee on hidden instances.

## Submission file

The file intended for submission is:

```text
factory-scheduler/bench-factory/adapters/myteam.py
```

It is a standalone, readable `Scheduler` implementation that uses only the
documented instance fields and submission callback.

## Approach

The solver is designed to produce a valid schedule immediately, then improve
it for as long as the time budget allows:

1. **Fast initial schedules** — builds several feasible dispatch schedules
   using different priorities, including earliest-start, due-date pressure,
   slack, remaining work, and weighted urgency.
2. **Portfolio selection** — submits the alternatives and retains the best
   valid machine ordering reported by the evaluator.
3. **Criticality-guided improvement** — focuses adjacent swaps around the
   operations contributing most to makespan and weighted tardiness.
4. **Bounded local search** — performs deterministic descent and controlled
   diversification with deadline checks, so it returns the best valid schedule
   within the allowed time.

## Run locally

From the repository root:

```bash
cd factory-scheduler/bench-factory
python self_check.py --adapter adapters.myteam:Scheduler
python run.py --adapter adapters.myteam:Scheduler --out report.json
```

`self_check.py` verifies basic feasibility on two public instances. `run.py`
executes the full supplied public suite and writes a JSON report.

## Repository layout

| Path | Purpose |
| --- | --- |
| `factory-scheduler/bench-factory/adapters/myteam.py` | Final scheduler submission |
| `factory-scheduler/bench-factory/README.md` | Challenge contract and scoring details |
| `factory-scheduler/bench-factory/run.py` | Full public benchmark runner |
| `factory-scheduler/bench-factory/self_check.py` | Fast validation check |
| `factory-scheduler/bench-factory/validator.py` | Canonical schedule validation and objective |

The repository also retains the other provided Phase 2 challenges for
reference. This submission targets **Factory Scheduler** only.

## Submission integrity

The solver is a general scheduling algorithm: it does not inspect private
evaluator state, mutate instances, access files or the network at runtime, or
hardcode public instance identities, seeds, digests, or precomputed answers.
The official organizer evaluation is authoritative.
