# Queue economics for a fraud and compliance review team

A working model of the decision every fraud team makes a few thousand times a day and
almost nobody prices: **who gets a human.**

```
python3 run.py      # stdlib only, no install, ~65 seconds
```

No customer data, no scraped data. A generated 24,000-alert queue with a deterministic
seed. The numbers are illustrative. The method, and the three failure modes it exposes,
are not.

---

## What it does

Treats the review queue as a constrained allocation problem rather than a ranking
problem. Each alert has three dispositions and three costs:

```
C_approve = p * v * LGF                                    fraud you eat
C_decline = (1-p) * (v*MARGIN + CHURN_P*LTV)  +  MU        good customer you burn
C_review  = t * LAMBDA + (1-ACC)*(C_approve + C_decline)   analyst time you spend
```

`LAMBDA` and `MU` are shadow prices, solved by bisection so the review set exactly fills
the analyst-hour budget and the decline set exactly fills the decline-rate budget.

- **LAMBDA** is what an analyst-minute is worth at the margin. It equals the wage when
  the team has slack and is strictly higher when the queue is oversubscribed.
- **MU** is what one slot in the decline budget is worth. Most fraud teams have a hard
  decline-rate ceiling set by the growth side of the business, and at a lot of shops it
  is the actually-binding constraint.

## Three things the run prints that are hard to see any other way

**1. A model can have a good AUC and still be unusable for any dollar decision.**

The generated score has AUC 0.866, which is a respectable production number, and the
0.0-0.1 band, which is 49% of the queue, over-states risk by 2.7x. Ranking metrics are
invariant to monotone transforms, so no dashboard built on AUC, precision, recall or KS
can see this. Every threshold expressed in dollars inherits the error, and inherits it
worst exactly where the volume is. Isotonic regression on a 40% holdout halves mean
absolute error against true risk and leaves AUC untouched.

**2. Fixing the probabilities without pricing the constraint makes things worse.**

This is the part I did not expect and it is the reason the shadow prices are in here at
all. Under a naive per-alert EV rule, the calibrated score *loses* to the uncalibrated
one. An over-confident score declines the alerts it cannot afford to review, which looks
like good recall while quietly spending a decline budget nobody priced. Calibrate the
score, leave the constraint unpriced, and you have made the system worse in a way no
offline metric will show you. Both constraints have to be in the objective.

**3. Alert recall is the wrong metric and moving it the wrong way is correct.**

Against a score-ranked baseline holding the same decline budget and the same six
analysts, total cost drops 54%. Alert recall drops 85.1% to 80.0%. Dollar recall rises
76.8% to 93.0%. The alerts the policy stops working are the cheap ones. Any queue
policy tuned on alert recall is being tuned to weight a $12 fraud like a $9,000 one.

## Two outputs that are arguments, not dashboards

**How many analysts you need.** Sweep the capacity constraint and read the shadow price.
While it sits above the wage the next analyst pays for themselves. In the generated run
the fifth analyst returns $2,882/month and the sixth returns $109, so the honest answer
is five. A capacity model that can only ever say "hire more" is not a model.

**What a wider decline budget is worth.** The fraud team and the growth team argue about
the decline rate and neither side brings a number. `MU` is the number: what one basis
point of decline rate is worth today, at today's model quality, in dollars.

## What it deliberately does not do

`SANCTIONS_NAME_FUZZY` returns $0.30 of fraud per $1 of analyst time and consumes 593
hours a month. It is also a sanctions rule, so it does not get switched off, and a
recommendation to switch it off would be the tell that whoever wrote this has never
worked next to a compliance function. It is pinned: always routed to a human regardless
of what the economics say, because the cost of not clearing it is regulatory rather than
financial and does not belong in a dollar objective. What the number is good for is
justifying engineering time on fuzzy-match precision, which is a real and fundable
project, instead of on the next model.

Constraints like that get pinned, not priced. The framework is only useful if it is
honest about which is which.

## Files

| | |
|---|---|
| `generate.py` | the alert stream. Draws true risk first, ground truth second, and a *distorted* score third, which is what makes the calibration section mean anything |
| `triage.py` | isotonic calibration, the cost model, the two-price bisection solver, the baselines, evaluation |
| `run.py` | the report |
| `results.json` | everything above, machine-readable |

Swap `LGF`, `MARGIN`, `CHURN_P`, `LTV`, `ANALYST_COST_PER_MIN` and `ANALYST_ACCURACY` at
the top of `triage.py` for real numbers and every output rescales.

## Where this comes from

I ran fraud detection and merchant risk models at Geidea, Saudi Arabia's largest fintech:
40,000+ merchants, 400,000+ POS terminals, over a million transactions a day, real-time
anomaly detection in production. The lesson that stuck was not about model architecture.
It was that the review queue is where model quality turns into money or fails to, and
that the queue almost never has an owner who thinks about it in dollars.

Syed Ahmed Basharat Ali. sabasharat.ali@gmail.com | basharat.net
