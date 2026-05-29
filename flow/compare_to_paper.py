"""
compare_to_paper.py — read eval_table.csv, generate markdown comparison vs paper.

Tables compared:
  - Paper Table 1: DiffPhyCon-lite FO-PC = 0.00037
  - Paper Table 25: γ-sweep on FO-PC (all γ in [0,1] give ~0.00037)

Usage:
    python flow/compare_to_paper.py \\
        --eval_csv results/paper_fopc_v2/eval_table.csv \\
        --out results/paper_fopc_v2/comparison.md
"""
import argparse
import csv
import os


# Paper Table 1 + Table 25 (FO-PC column)
PAPER_TABLE1_FOPC = 0.00037   # DiffPhyCon-lite (joint only, γ=1)
PAPER_TABLE25 = {              # γ-sweep on FO-PC, J_actual
    0.0: 0.00038,
    0.3: 0.00037,
    0.5: 0.00037,
    0.7: 0.00037,
    1.0: 0.00037,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_csv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = []
    with open(args.eval_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "variant": r["variant"],
                "gamma": float(r["gamma"]),
                "J_mean": float(r["J_mean"]),
                "J_std": float(r["J_std"]),
                "E_mean": float(r["E_mean"]),
            })

    lines = []
    lines.append("# FM Burgers FO-PC — our results vs paper\n")
    lines.append("Eval config: N_TEST=50, n_steps=1000, EMA weights, FOPC w-mask.\n")
    lines.append(f"Paper baseline (Table 1, DiffPhyCon-lite FO-PC): **J = {PAPER_TABLE1_FOPC}**\n")

    # ---- Table A: Best J per variant at γ=1 (DiffPhyCon-lite analogue) ----
    lines.append("\n## A. DiffPhyCon-lite analogue (joint only, γ=1)\n")
    lines.append("| Variant | J (ours) | J_std | E (ours) | vs paper |")
    lines.append("|:---|---:|---:|---:|---:|")
    lines.append(f"| paper DiffPhyCon-lite | {PAPER_TABLE1_FOPC} | — | — | 1.0× |")
    for r in rows:
        if abs(r["gamma"] - 1.0) < 1e-8:
            ratio = r["J_mean"] / PAPER_TABLE1_FOPC
            lines.append(
                f"| {r['variant']} | {r['J_mean']:.5f} | {r['J_std']:.5f} | "
                f"{r['E_mean']:.1f} | {ratio:.1f}× |"
            )

    # ---- Table B: γ-sweep replicating paper Table 25 ----
    lines.append("\n## B. γ-sweep (replicating paper Table 25, FO-PC)\n")
    variants = sorted({r["variant"] for r in rows})
    gammas = sorted({r["gamma"] for r in rows})
    header = "| γ | paper Table 25 |" + "".join(f" {v} J |" for v in variants)
    sep = "|:---:|:---:|" + "".join(":---:|" for _ in variants)
    lines.append(header)
    lines.append(sep)
    for g in gammas:
        paper_val = PAPER_TABLE25.get(g, "—")
        paper_str = f"{paper_val}" if paper_val != "—" else "—"
        row_str = f"| {g} | {paper_str} |"
        for v in variants:
            match = [r for r in rows if r["variant"] == v and abs(r["gamma"] - g) < 1e-8]
            if match:
                row_str += f" {match[0]['J_mean']:.5f} |"
            else:
                row_str += " — |"
        lines.append(row_str)

    # ---- Section C: high-level findings ----
    lines.append("\n## C. Key findings\n")
    g1_rows = [r for r in rows if abs(r["gamma"] - 1.0) < 1e-8]
    if len(g1_rows) >= 2:
        vanilla = next((r for r in g1_rows if r["variant"] == "vanilla"), None)
        ot = next((r for r in g1_rows if r["variant"] == "ot"), None)
        if vanilla and ot:
            lines.append(f"- **vanilla vs OT at γ=1**: vanilla J = {vanilla['J_mean']:.5f}, "
                         f"OT J = {ot['J_mean']:.5f} "
                         f"(OT {'wins' if ot['J_mean'] < vanilla['J_mean'] else 'loses'} on J).")
            lines.append(f"- **Energy**: vanilla E = {vanilla['E_mean']:.1f}, "
                         f"OT E = {ot['E_mean']:.1f} "
                         f"(OT {'wins' if ot['E_mean'] < vanilla['E_mean'] else 'loses'} on E).")

    if len(g1_rows):
        best_J = min(r["J_mean"] for r in g1_rows)
        ratio = best_J / PAPER_TABLE1_FOPC
        verdict = "✅ MATCH" if ratio < 2 else "⚠️ close" if ratio < 5 else "❌ gap"
        lines.append(f"- **Best vs paper**: {best_J:.5f} vs paper {PAPER_TABLE1_FOPC} = {ratio:.1f}× ({verdict}).")

    # γ-flat check
    vanilla_rows = [r for r in rows if r["variant"] == "vanilla"]
    if len(vanilla_rows) >= 2:
        Js = [r["J_mean"] for r in vanilla_rows]
        spread = (max(Js) - min(Js)) / (sum(Js) / len(Js))
        if spread < 0.05:
            lines.append(f"- **γ-flat replicated**: J varies <{spread*100:.1f}% across γ "
                         f"(paper L.1: γ has near-zero effect on J on FO-PC).")
        else:
            lines.append(f"- **γ-flat NOT replicated**: J varies {spread*100:.1f}% across γ "
                         f"(unexpected — paper L.1 says <1%).")

    out_text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(out_text)
    print(f"💾 wrote {args.out}")
    print("\n" + "=" * 60)
    print(out_text)


if __name__ == "__main__":
    main()
