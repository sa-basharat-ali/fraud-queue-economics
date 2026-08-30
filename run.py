#!/usr/bin/env python3
"""Fraud & compliance queue: expected-value triage under two hard constraints.

    python3 run.py        stdlib only, no install, ~65s

Everything below is generated. No customer data, no scraped data, deterministic seed.
The point is the method and the failure modes it exposes, not the numbers.
"""
import json, random, os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import generate, triage as T

money = lambda x: f"${x:,.0f}"
line  = lambda c="=", n=84: print(c * n)

alerts = generate.build(40000, 30)
random.seed(7)
random.shuffle(alerts)
cut = int(len(alerts) * 0.40)
fit_set, eval_set = alerts[:cut], alerts[cut:]
eval_set.sort(key=lambda a: a["ts"])
days = len({a["ts"][:10] for a in eval_set})
at_risk = sum(a["amount_usd"] for a in eval_set if a["is_fraud"])
demand = sum(a["review_minutes_est"] for a in eval_set) / days

print()
line()
print("  FRAUD & COMPLIANCE QUEUE: EXPECTED-VALUE TRIAGE UNDER TWO CONSTRAINTS")
line()
print(f"  {len(eval_set):,} alerts / {days} days   "
      f"{sum(a['is_fraud'] for a in eval_set)/len(eval_set):.1%} of the queue is real fraud   "
      f"score AUC {generate.auc(eval_set):.3f}")
print(f"  {money(at_risk)} of fraud sitting in the queue")
print(f"  capacity {T.ANALYST_MINUTES_DAY/60:.0f} analyst-hours/day against "
      f"{demand/60:.0f} hours of demand  ->  {demand/T.ANALYST_MINUTES_DAY:.1f}x oversubscribed")
print(f"  decline budget {T.DECLINE_RATE_BUDGET:.1%} of the queue")

# ---- 1. calibration ---------------------------------------------------------
print("\n\n1. THE SCORE RANKS WELL. IT IS NOT A PROBABILITY.\n")
print(f"   {'score band':<13}{'alerts':>8}{'score says':>13}{'actually is':>14}{'over by':>10}")
print("   " + "-" * 58)
worst = None
for r in T.calibration_table(eval_set):
    lift = f"{r['lift']:.2f}x" if r["lift"] else "-"
    flag = ""
    if r["lift"] and r["n"] > 400 and r["lift"] > 1.4:
        flag = "  <--"
        if worst is None or r["n"] > worst["n"]:
            worst = r
    print(f"   {r['bucket']:<13}{r['n']:>8,}{r['mean_predicted']:>13.3f}"
          f"{r['observed']:>14.3f}{lift:>10}{flag}")

cal = T.fit_isotonic(fit_set)
print(f"\n   AUC {generate.auc(eval_set):.3f} says the ordering is good, and it is. But the "
      f"{worst['bucket']} band holds")
print(f"   {worst['n']:,} alerts ({worst['n']/len(eval_set):.0%} of the queue) and over-states "
      f"risk by {worst['lift']:.1f}x. Ranking metrics")
print(f"   cannot see this. Every dollar threshold built on the raw score inherits the")
print(f"   error, and inherits it worst exactly where the volume is.")

mae_raw = st.mean(abs(a["model_score"] - a["_true_p"]) for a in eval_set)
mae_cal = st.mean(abs(cal(a["model_score"]) - a["_true_p"]) for a in eval_set)
print(f"\n   Isotonic regression fit on a 40% holdout, {cal.n_blocks} blocks, applied to the rest:")
print(f"   mean absolute error against true risk {mae_raw:.4f} -> {mae_cal:.4f}. "
      f"Ordering unchanged,")
print(f"   so AUC is unchanged. This is invisible to every ranking metric on the dashboard.")
print(f"\n   {'calibrated band':<17}{'alerts':>8}{'says':>10}{'actually is':>14}")
print("   " + "-" * 49)
for r in T.calibration_table(eval_set, lambda a: cal(a["model_score"])):
    print(f"   {r['bucket']:<17}{r['n']:>8,}{r['mean_predicted']:>10.3f}{r['observed']:>14.3f}")

# ---- 2. policies ------------------------------------------------------------
oracle_set = [dict(a, model_score=a["_true_p"]) for a in eval_set]
policies = [
    T.evaluate(T.triage_threshold(eval_set, order="fifo"),  "arrival order + score cutoffs"),
    T.evaluate(T.triage_threshold(eval_set, order="score"), "score-ranked + score cutoffs"),
    T.evaluate(T.triage(eval_set, lambda s: s),             "EV triage, raw score"),
    T.evaluate(T.triage(eval_set, cal),                     "EV triage, calibrated"),
    T.evaluate(T.triage(oracle_set, lambda s: s),           "ceiling: EV on true risk"),
]
base, best, ceiling = policies[1], policies[3], policies[4]

print("\n\n2. WHAT THE SAME QUEUE COSTS UNDER FIVE POLICIES\n")
print(f"   {'policy':<32}{'fraud loss':>12}{'false decl':>12}{'analyst':>10}{'TOTAL':>12}")
print("   " + "-" * 81)
for r in policies:
    mark = "  <-- today" if r is base else ("  <-- proposed" if r is best else "")
    print(f"   {r['policy']:<32}{money(r['fraud_loss']):>12}"
          f"{money(r['false_decline_cost']):>12}{money(r['analyst_cost']):>10}"
          f"{money(r['total_cost']):>12}{mark}")
print(f"\n   {'policy':<32}{'analyst-hrs':>12}{'reviewed':>11}{'alert recall':>14}"
      f"{'$ recall':>11}")
print("   " + "-" * 81)
for r in policies:
    print(f"   {r['policy']:<32}{r['analyst_hours']:>12,.0f}{r['n_review']:>11,}"
          f"{r['recall']:>14.1%}{r['value_recall']:>11.1%}")

saved = base["total_cost"] - best["total_cost"]
hours = base["analyst_hours"] - best["analyst_hours"]
print(f"\n   Same six analysts, same model, same alerts, same {T.DECLINE_RATE_BUDGET:.0%} decline budget.")
print(f"   Total cost {money(base['total_cost'])} -> {money(best['total_cost'])} "
      f"({saved/base['total_cost']:.0%}), {money(saved*365/days)} annualised,")
print(f"   on {hours:,.0f} fewer analyst-hours a month.")
print(f"\n   Alert recall goes DOWN, {base['recall']:.1%} -> {best['recall']:.1%}, and that is the "
      f"point. Alert recall weights a")
print(f"   $12 fraud the same as a $9,000 one. Dollar recall goes "
      f"{base['value_recall']:.1%} -> {best['value_recall']:.1%}. The alerts")
print(f"   the policy stops working are the cheap ones.")
gap = base['total_cost'] - ceiling['total_cost']
print(f"\n   The last row is the ceiling: what an oracle with perfect per-alert risk would")
print(f"   spend. The proposed policy captures {(saved/gap):.0%} of the available gain, so the")
print(f"   remaining headroom is in the model, not in the queue policy.")

# ---- 3. how many analysts, and how big a decline budget --------------------
print("\n\n3. HOW MANY ANALYSTS DO YOU ACTUALLY NEED\n")
print(f"   {'analysts':>9}{'hrs avail':>11}{'hrs used':>10}{'$/min at margin':>17}"
      f"{'TOTAL cost':>13}{'marginal':>11}")
print("   " + "-" * 71)
prev = None
for k in (2, 3, 4, 5, 6, 7, 8, 10):
    budget = k * 8 * 60 * 0.72
    d, pr = T.triage(eval_set, cal, minutes_budget=budget, return_prices=True)
    r = T.evaluate(d, f"{k} analysts")
    lam = st.mean(p["lambda_per_min"] for p in pr.values())
    marg = "" if prev is None else money(prev - r["total_cost"])
    binding = "  <-- slack" if lam <= T.ANALYST_COST_PER_MIN * 1.001 else ""
    print(f"   {k:>9}{budget*days/60:>11,.0f}{r['analyst_hours']:>10,.0f}"
          f"{lam:>17.2f}{money(r['total_cost']):>13}{marg:>11}{binding}")
    prev = r["total_cost"]
print(f"\n   Read the $/min column. While it sits above the ${T.ANALYST_COST_PER_MIN:.2f} wage the "
      f"queue is capacity-bound and")
print(f"   the next analyst pays for themselves. The moment it drops to the wage the")
print(f"   constraint has gone slack and the next hire buys nothing: the policy is already")
print(f"   declining to review the alerts that are not worth reviewing.")
print(f"\n   That is the headcount conversation with a number attached, which is the thing")
print(f"   headcount arguments normally lack. It runs the other way too, and the honest")
print(f"   version of this analysis is the one that tells you when to stop hiring.")

print("\n\n   WHAT A WIDER DECLINE BUDGET IS WORTH\n")
print(f"   {'decline budget':>15}{'$/decline slot':>17}{'TOTAL cost':>13}{'vs 3.0%':>11}")
print("   " + "-" * 56)
b30 = None
for rate in (0.015, 0.020, 0.030, 0.040, 0.060, 0.100):
    d, pr = T.triage(eval_set, cal, decline_rate=rate, return_prices=True)
    r = T.evaluate(d, "")
    mu = st.mean(p["mu_per_decline"] for p in pr.values())
    if abs(rate - T.DECLINE_RATE_BUDGET) < 1e-9:
        b30 = r["total_cost"]
    delta = "" if b30 is None else money(r["total_cost"] - b30)
    print(f"   {rate:>15.1%}{mu:>17.2f}{money(r['total_cost']):>13}{delta:>11}")
print(f"\n   Every fraud team argues about the decline rate with the growth team and neither")
print(f"   side has a number. This is the number. Each row is a defensible position and the")
print(f"   $/decline-slot column is what one basis point of decline rate is worth today.")

# ---- 4. rules ---------------------------------------------------------------
print("\n\n4. WHICH RULES ARE EATING THE QUEUE\n")
print(f"   {'rule':<26}{'alerts':>7}{'prec':>7}{'hours':>7}{'review $':>10}"
      f"{'fraud $':>10}{'$ back per $':>14}")
print("   " + "-" * 81)
econ = T.rule_economics(eval_set)
for r in econ:
    print(f"   {r['rule_id']:<26}{r['alerts']:>7,}{r['precision']:>7.1%}"
          f"{r['analyst_hours']:>7,.0f}{money(r['review_cost']):>10}"
          f"{money(r['fraud_usd_surfaced']):>10}{r['roi']:>14.2f}")
dead = [r for r in econ if r["roi"] < 1.5]
if dead:
    h = sum(r["analyst_hours"] for r in dead)
    c = sum(r["review_cost"] for r in dead)
    names = ", ".join(r["rule_id"] for r in dead)
    print(f"\n   {names} returns under $1.50 of fraud per $1 of")
    print(f"   analyst time: {h:,.0f} hours and {money(c)} over {days} days.")
    print(f"\n   And it is a sanctions rule, so no, you cannot stop reviewing it. That is")
    print(f"   exactly why it belongs in this table. A rule you are legally required to")
    print(f"   clear is not a candidate for deletion, it is a candidate for automation, and")
    print(f"   this is the number that justifies spending engineering time on fuzzy-match")
    print(f"   precision instead of on the next model. The EV policy will keep routing")
    print(f"   these to humans no matter what the economics say, because the cost of not")
    print(f"   clearing them is regulatory rather than financial and does not belong in a")
    print(f"   dollar objective. Constraints like that get pinned, not priced.")

json.dump({"policies": policies, "rules": econ, "days": days,
           "annualised_saving": saved * 365 / days, "auc": generate.auc(eval_set),
           "mae_raw": mae_raw, "mae_calibrated": mae_cal},
          open("results.json", "w"), indent=2, default=float)
print(f"\n   -> results.json\n")
line()
