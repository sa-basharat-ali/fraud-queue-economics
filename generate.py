"""Synthetic but realistically-shaped alert stream for a fraud/compliance review queue.

Construction (this ordering is what makes the calibration section meaningful):
  1. each alert gets a TRUE risk p drawn from a rule-specific Beta
  2. ground truth is Bernoulli(p)
  3. the model score is a *distorted* view of p:  score = p**0.72 * lognormal noise

Step 3 is the single most common defect in a production risk score. Gradient-boosted
ensembles trained with class weighting systematically inflate low probabilities, so
score and probability are monotone-ish but not equal. Every dollar threshold set on
the raw score is then wrong, and wrong hardest in the densest part of the queue.

Shapes calibrated to publicly reported card-not-present norms: alert-level precision
in the single digits to low teens, log-normal ticket sizes with a long right tail.
No customer data. Deterministic seed.
"""
import math, random, json
from datetime import datetime, timedelta

SEED = 20260829
random.seed(SEED)

# rule_id: (share_of_queue, mean_true_risk, beta_concentration, avg_review_minutes)
RULES = {
    "VELOCITY_CARD_24H":       (0.22, 0.031,  9.0,  6.0),
    "DEVICE_REUSE_MULTI_ACCT": (0.14, 0.181,  4.5,  9.0),
    "BEHAVIOR_PASTE_PAN":      (0.09, 0.264,  3.6,  7.5),
    "GEO_IP_BILLING_MISMATCH": (0.16, 0.048, 11.0,  5.5),
    "EMAIL_AGE_LT_7D":         (0.11, 0.072,  8.0,  4.5),
    "AMOUNT_ZSCORE_GT_3":      (0.08, 0.113,  5.0,  8.0),
    "SANCTIONS_NAME_FUZZY":    (0.07, 0.019, 14.0, 22.0),   # compliance: slow and noisy
    "BIN_COUNTRY_HIGH_RISK":   (0.06, 0.055, 10.0,  5.0),
    "SESSION_EMULATOR_SIG":    (0.04, 0.412,  3.0, 11.0),
    "CHARGEBACK_LINKED_ENT":   (0.03, 0.531,  2.6, 12.0),
}

DISTORTION = 0.72   # score = p ** DISTORTION  -> low probabilities read too high
NOISE_SIGMA = 0.30  # ranking noise: the score is not a perfect ordering of true risk


def _amount():
    return round(min(math.exp(random.gauss(4.45, 1.35)), 42000.0), 2)


def build(n=12000, days=14):
    rules = list(RULES)
    weights = [RULES[r][0] for r in rules]
    t0 = datetime(2026, 8, 1)
    out = []
    for i in range(n):
        rule = random.choices(rules, weights=weights, k=1)[0]
        _, mean_p, conc, mins = RULES[rule]
        a = max(mean_p * conc, 0.05)
        b = max((1 - mean_p) * conc, 0.05)
        true_p = min(max(random.betavariate(a, b), 0.001), 0.995)
        is_fraud = random.random() < true_p

        score = (true_p ** DISTORTION) * math.exp(random.gauss(0, NOISE_SIGMA))
        score = round(min(max(score, 0.001), 0.999), 4)

        amount = _amount()
        if is_fraud:                       # fraud skews to larger tickets
            amount = round(min(amount * random.uniform(1.3, 2.6), 42000.0), 2)

        day = random.randrange(days)       # arrivals cluster in business hours
        hour = int(min(max(random.gauss(13.5, 3.4), 0), 23.99))
        ts = t0 + timedelta(days=day, hours=hour, minutes=random.randrange(60))

        out.append({
            "alert_id": f"A{i:06d}",
            "ts": ts.isoformat(timespec="seconds"),
            "rule_id": rule,
            "model_score": score,
            "amount_usd": amount,
            "review_minutes_est": round(max(random.gauss(mins, mins * 0.28), 1.0), 1),
            "customer_tenure_days": max(0, int(random.expovariate(1 / 240))),
            "is_fraud": int(is_fraud),     # ground truth, evaluation only
            "_true_p": round(true_p, 5),   # diagnostics only; never used by the policy
        })
    out.sort(key=lambda r: r["ts"])
    return out


def auc(alerts):
    pos = sorted(a["model_score"] for a in alerts if a["is_fraud"])
    neg = sorted(a["model_score"] for a in alerts if not a["is_fraud"])
    if not pos or not neg:
        return float("nan")
    merged = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank, i, total = 0.0, 0, 0.0
    while i < len(merged):
        j = i
        while j < len(merged) and merged[j][0] == merged[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        total += sum(avg_rank for k in range(i, j) if merged[k][1] == 1)
        i = j
    return (total - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


if __name__ == "__main__":
    alerts = build()
    with open("alerts.json", "w") as f:
        json.dump(alerts, f)
    fr = sum(a["is_fraud"] for a in alerts)
    print(f"{len(alerts)} alerts | {fr} fraud ({fr/len(alerts):.2%}) | "
          f"${sum(a['amount_usd'] for a in alerts if a['is_fraud']):,.0f} at risk | "
          f"score AUC {auc(alerts):.3f}")
