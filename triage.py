"""Expected-value triage for a fraud / compliance review queue.

A review queue is not a ranking problem. It is a constrained allocation problem with
two scarce resources, and every alert has three dispositions with three costs:

    C_approve(i) = p * v * LGF                                  fraud we eat
    C_decline(i) = (1-p) * (v*MARGIN + CHURN_P*LTV)  + MU       good customer we burn
    C_review(i)  = t * LAMBDA + (1-ACC)*(C_approve + C_decline) analyst time we spend

LAMBDA and MU are shadow prices, not costs.

  LAMBDA  dollars per analyst-minute at the margin. Equals the wage when the team has
          slack. When the queue is oversubscribed it is strictly higher, and the gap
          is the price of the hours you do not have.

  MU      dollars per auto-decline. Fraud teams do not get to decline as much as the
          math wants. There is a decline-rate budget set by the business, and it is
          the binding constraint at most shops. MU is what one slot in that budget is
          worth.

Both are solved by bisection so the review set exactly fills the analyst budget and
the decline set exactly fills the decline budget.

Why this matters, and it is the whole point of the exercise: without MU, better
probabilities make the policy *worse*. An over-confident score declines the alerts it
cannot afford to review and looks great on recall, while quietly spending a decline
budget nobody priced. Calibrate the score without pricing the constraint and you have
made the system worse in a way no offline metric will show you.
"""
from collections import defaultdict

# --- economics (swap for your real numbers; every output scales linearly) ---
LGF                  = 1.00   # loss given fraud, as a fraction of the ticket
MARGIN               = 0.028  # revenue lost on a good transaction you decline
CHURN_P              = 0.085  # a wrongly declined customer sometimes never comes back
LTV                  = 430.0
ANALYST_COST_PER_MIN = 1.15   # $69/hr fully loaded
ANALYST_ACCURACY     = 0.94
ANALYST_MINUTES_DAY  = 6 * 8 * 60 * 0.72   # 6 analysts, 8h day, 72% utilisation
DECLINE_RATE_BUDGET  = 0.030               # auto-decline at most 3% of the queue


def cost_approve(p, v):
    return p * v * LGF

def cost_decline(p, v):
    return (1 - p) * (v * MARGIN + CHURN_P * LTV)

def residual_after_review(p, v):
    return (1 - ANALYST_ACCURACY) * (cost_approve(p, v) + cost_decline(p, v))


# ---------------------------------------------------------------- calibration
def calibration_table(alerts, score_key="model_score", bins=10):
    buckets = defaultdict(lambda: [0, 0.0, 0])
    for a in alerts:
        s = a[score_key] if isinstance(score_key, str) else score_key(a)
        b = min(int(s * bins), bins - 1)
        buckets[b][0] += 1
        buckets[b][1] += s
        buckets[b][2] += a["is_fraud"]
    rows = []
    for b in sorted(buckets):
        n, sp, nf = buckets[b]
        rows.append({"bucket": f"{b/bins:.1f}-{(b+1)/bins:.1f}", "n": n,
                     "mean_predicted": sp / n, "observed": nf / n,
                     "lift": (sp / n) / (nf / n) if nf else None})
    return rows


def fit_isotonic(alerts, min_block=250):
    """Pool-adjacent-violators isotonic regression: raw score -> probability.
    Monotone, non-parametric, one pass, no dependencies.

    Two details that matter and are easy to get wrong:

    1. Blocks are looked up by containment, not by nearest-knot-below. The obvious
       binary search returns the previous block and shifts every probability down.

    2. Tail blocks are merged until each holds at least `min_block` observations.
       PAVA drives a block of three alerts to exactly 0.0 or 1.0, which is an
       artefact of block size rather than evidence. The tempting fix -- shrink each
       block toward the global base rate -- is worse than the disease: it lifts the
       low end (which is most of a fraud queue) toward the mean and the monotonicity
       repair then propagates that lift forward. Merging fixes the variance without
       touching the level.
    """
    pts = sorted((a["model_score"], float(a["is_fraud"])) for a in alerts)

    def pava(seq):                       # seq: [sum_y, weight, x_lo, x_hi]
        stack = []
        for blk in seq:
            stack.append(list(blk))
            while len(stack) > 1 and stack[-2][0] / stack[-2][1] > stack[-1][0] / stack[-1][1]:
                sy2, w2, lo2, hi2 = stack.pop()
                sy1, w1, lo1, hi1 = stack.pop()
                stack.append([sy1 + sy2, w1 + w2, lo1, hi2])
        return stack

    stack = pava([[y, 1.0, x, x] for x, y in pts])
    while len(stack) > 1:
        i = min(range(len(stack)), key=lambda k: stack[k][1])
        if stack[i][1] >= min_block:
            break
        j = i - 1 if i == len(stack) - 1 else (
            i + 1 if i == 0 else (i - 1 if stack[i - 1][1] <= stack[i + 1][1] else i + 1))
        lo, hi = min(i, j), max(i, j)
        a1, a2 = stack[lo], stack[hi]
        stack[lo:hi + 1] = [[a1[0] + a2[0], a1[1] + a2[1], a1[2], a2[3]]]
        stack = pava(stack)

    blocks = [(lo, hi, sy / w) for sy, w, lo, hi in stack]
    los = [b[0] for b in blocks]

    def apply(s):
        if s <= blocks[0][1]:
            return blocks[0][2]
        if s >= blocks[-1][0]:
            return blocks[-1][2]
        lo, hi = 0, len(blocks) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if los[mid] <= s: lo = mid
            else:             hi = mid - 1
        b = blocks[lo]
        if s <= b[1] or lo + 1 >= len(blocks):
            return b[2]
        nxt = blocks[lo + 1]                    # interpolate across the gap
        span = nxt[0] - b[1]
        return b[2] if span <= 0 else b[2] + (s - b[1]) / span * (nxt[2] - b[2])

    apply.n_blocks = len(blocks)
    return apply


# ------------------------------------------------------------------ the policy
def _dispose(pool, lam, mu):
    """Cheapest disposition per alert at the given shadow prices."""
    out = []
    for x in pool:
        c_app = x["c_app"]
        c_dec = x["c_dec"] + mu
        c_rev = x["t"] * lam + x["residual"]
        best = min(c_app, c_dec, c_rev)
        out.append("approve" if best == c_app else ("decline" if best == c_dec else "review"))
    return out


def _solve_prices(pool, minutes_budget, decline_budget, iters=14):
    """Coordinate bisection on (LAMBDA, MU). Raising either price monotonically
    shrinks the set that consumes that resource, so each inner solve is a clean
    bisection and the outer loop converges in a handful of passes."""
    lam, mu = ANALYST_COST_PER_MIN, 0.0
    for _ in range(iters):
        lo, hi = ANALYST_COST_PER_MIN, ANALYST_COST_PER_MIN
        used = sum(x["t"] for x, d in zip(pool, _dispose(pool, lo, mu)) if d == "review")
        if used > minutes_budget:
            hi = lo
            while hi < 1e7:
                hi *= 2
                if sum(x["t"] for x, d in zip(pool, _dispose(pool, hi, mu)) if d == "review") <= minutes_budget:
                    break
            for _ in range(48):
                mid = (lo + hi) / 2
                u = sum(x["t"] for x, d in zip(pool, _dispose(pool, mid, mu)) if d == "review")
                lo, hi = (mid, hi) if u > minutes_budget else (lo, mid)
            lam = hi
        else:
            lam = lo

        lo, hi = 0.0, 0.0
        n_dec = sum(1 for d in _dispose(pool, lam, 0.0) if d == "decline")
        if n_dec > decline_budget:
            hi = 1.0
            while hi < 1e7:
                hi *= 2
                if sum(1 for d in _dispose(pool, lam, hi) if d == "decline") <= decline_budget:
                    break
            for _ in range(48):
                mid = (lo + hi) / 2
                if sum(1 for d in _dispose(pool, lam, mid) if d == "decline") > decline_budget:
                    lo = mid
                else:
                    hi = mid
            mu = hi
        else:
            mu = 0.0
    return lam, mu


def triage(alerts, prob_fn, minutes_budget=ANALYST_MINUTES_DAY,
           decline_rate=DECLINE_RATE_BUDGET, return_prices=False):
    by_day = defaultdict(list)
    for a in alerts:
        by_day[a["ts"][:10]].append(a)

    decided, prices = [], {}
    for day in sorted(by_day):
        pool = []
        for a in by_day[day]:
            p = prob_fn(a["model_score"])
            v, t = a["amount_usd"], a["review_minutes_est"]
            pool.append({"a": a, "p": p, "t": t,
                         "c_app": cost_approve(p, v),
                         "c_dec": cost_decline(p, v),
                         "residual": residual_after_review(p, v)})
        lam, mu = _solve_prices(pool, minutes_budget, decline_rate * len(pool))
        prices[day] = {"lambda_per_min": lam, "mu_per_decline": mu}
        for x, action in zip(pool, _dispose(pool, lam, mu)):
            decided.append({"a": x["a"], "p": x["p"], "day": day, "action": action})
    return (decided, prices) if return_prices else decided


def triage_threshold(alerts, minutes_budget=ANALYST_MINUTES_DAY,
                     decline_rate=DECLINE_RATE_BUDGET, order="score"):
    """The two baselines that between them cover what most teams actually run.

    order="fifo"  work the queue in arrival order   (no prioritisation)
    order="score" work highest score first          (the competent status quo)

    Both honour the same decline budget: each day the top decline_rate share by score
    is auto-declined. Whatever is left is reviewed until the hours run out, and the
    rest is approved. Neither baseline looks at the dollars on the alert or at how
    long the review will take. That is the only thing being tested here.
    """
    by_day = defaultdict(list)
    for a in alerts:
        by_day[a["ts"][:10]].append(a)
    decided = []
    for day in sorted(by_day):
        pool = by_day[day]
        k = int(decline_rate * len(pool))
        declined = {a["alert_id"] for a in
                    sorted(pool, key=lambda r: -r["model_score"])[:k]}
        rest = [a for a in pool if a["alert_id"] not in declined]
        rest.sort(key=(lambda r: r["ts"]) if order == "fifo"
                  else (lambda r: -r["model_score"]))
        spent = 0.0
        for a in pool:
            if a["alert_id"] in declined:
                decided.append({"a": a, "day": day, "action": "decline"})
        for a in rest:
            t = a["review_minutes_est"]
            if spent + t <= minutes_budget:
                spent += t
                action = "review"
            else:
                action = "approve"
            decided.append({"a": a, "day": day, "action": action})
    return decided


# ------------------------------------------------------------------ evaluation
def evaluate(decided, label):
    fraud_loss = fd_cost = analyst_cost = 0.0
    minutes = 0.0
    caught = missed = 0
    caught_usd = total_usd = 0.0
    n = {"review": 0, "approve": 0, "decline": 0}
    for d in decided:
        a = d["a"]
        v, f = a["amount_usd"], a["is_fraud"]
        act = d["action"]
        n[act] += 1
        if f:
            total_usd += v
        if act == "review":
            minutes += a["review_minutes_est"]
            analyst_cost += a["review_minutes_est"] * ANALYST_COST_PER_MIN
            if f:
                caught += 1
                caught_usd += ANALYST_ACCURACY * v
                fraud_loss += (1 - ANALYST_ACCURACY) * v * LGF
            else:
                fd_cost += (1 - ANALYST_ACCURACY) * (v * MARGIN + CHURN_P * LTV)
        elif act == "approve":
            if f:
                missed += 1
                fraud_loss += v * LGF
        else:
            if f:
                caught += 1
                caught_usd += v
            else:
                fd_cost += v * MARGIN + CHURN_P * LTV
    total = len(decided)
    return {"policy": label, "fraud_loss": fraud_loss, "false_decline_cost": fd_cost,
            "analyst_cost": analyst_cost,
            "total_cost": fraud_loss + fd_cost + analyst_cost,
            "analyst_hours": minutes / 60, "n_review": n["review"],
            "n_auto_approve": n["approve"], "n_auto_decline": n["decline"],
            "auto_decline_rate": n["decline"] / total,
            "fraud_caught": caught, "fraud_missed": missed,
            "recall": caught / max(caught + missed, 1),
            "value_recall": caught_usd / max(total_usd, 1e-9)}


# ------------------------------------------------------- rule-level economics
PINNED_RULES = {"SANCTIONS_NAME_FUZZY"}   # regulatory: always routed to a human


def rule_economics(alerts):
    agg = defaultdict(lambda: {"n": 0, "fraud": 0, "minutes": 0.0, "fraud_usd": 0.0})
    for a in alerts:
        r = agg[a["rule_id"]]
        r["n"] += 1
        r["fraud"] += a["is_fraud"]
        r["minutes"] += a["review_minutes_est"]
        r["fraud_usd"] += a["amount_usd"] * a["is_fraud"]
    rows = []
    for rule, r in agg.items():
        review_cost = r["minutes"] * ANALYST_COST_PER_MIN
        rows.append({"rule_id": rule, "alerts": r["n"], "precision": r["fraud"] / r["n"],
                     "analyst_hours": r["minutes"] / 60, "review_cost": review_cost,
                     "fraud_usd_surfaced": r["fraud_usd"],
                     "roi": r["fraud_usd"] / review_cost if review_cost else 0.0})
    return sorted(rows, key=lambda x: x["roi"])
