from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
SHOWCASE_DIR = ROOT / "showcase"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)
    output = json.loads((PROCESSED_DIR / "model_output.json").read_text(encoding="utf-8"))
    components = pd.read_csv(PROCESSED_DIR / "component_forecast.csv")
    scenarios = pd.read_csv(PROCESSED_DIR / "model_scenarios.csv")
    comparison = pd.read_csv(PROCESSED_DIR / "market_comparison.csv")
    quality = pd.read_csv(PROCESSED_DIR / "data_quality_report.csv")
    headline = output["headline_forecast"]

    scenario_rows = "\n".join(
        f"<tr><td>{row.scenario}</td><td>{pct(row.gallrein_win_probability)}</td><td>{pct(row.massie_win_probability)}</td><td>{row.mean_massie_margin_points:.1f}</td></tr>"
        for row in scenarios.itertuples()
    )
    component_rows = "\n".join(
        f"<tr><td>{row.component}</td><td>{pct(row.gallrein_win_probability)}</td><td>{pct(row.massie_win_probability)}</td><td>{row.massie_margin_points:.1f}</td><td>{pct(row.model_weight)}</td></tr>"
        for row in components.itertuples()
    )
    market_rows = "\n".join(
        f"<tr><td>{row.platform}</td><td>{pct(row.gallrein_market_probability)}</td><td>{pct(row.gallrein_fair_probability)}</td><td>{row.gallrein_edge_points:.1f}</td></tr>"
        for row in comparison.itertuples()
    )
    quality_rows = "\n".join(
        f"<tr><td>{row.signal_group}</td><td>{row.status}</td><td>{row.notes}</td></tr>"
        for row in quality.itertuples()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KY-04 Republican Primary Forecast</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Arial, sans-serif;
      --ink: #17202a;
      --muted: #5d6673;
      --line: #d9dee7;
      --blue: #1f5a8a;
      --red: #9f2b2b;
      --paper: #f7f8fa;
      --white: #fff;
    }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); }}
    header {{ padding: 28px 36px 18px; background: var(--white); border-bottom: 1px solid var(--line); }}
    main {{ padding: 24px 36px 40px; max-width: 1120px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ font-size: 17px; margin: 26px 0 10px; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .metric {{ background: var(--white); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .value {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; background: var(--white); border: 1px solid var(--line); }}
    th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--line); font-size: 14px; }}
    th {{ background: #eef1f5; }}
    .note {{ max-width: 850px; line-height: 1.5; }}
    @media (max-width: 760px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      .grid {{ grid-template-columns: 1fr; }}
      table {{ font-size: 13px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>KY-04 Republican Primary Forecast</h1>
    <div class="muted">Pre-election fair odds as of {output["as_of"]}. Official vote totals are excluded.</div>
  </header>
  <main>
    <section class="grid">
      <div class="metric"><div class="label">Ed Gallrein Fair Odds</div><div class="value">{pct(headline["gallrein_fair_probability"])}</div></div>
      <div class="metric"><div class="label">Thomas Massie Fair Odds</div><div class="value">{pct(headline["massie_fair_probability"])}</div></div>
      <div class="metric"><div class="label">Mean Massie Margin</div><div class="value">{headline["mean_massie_margin_points"]:.1f}</div></div>
    </section>
    <p class="note muted">This is an auditable pre-result ensemble, not financial advice. Markets are included only as a bounded component, while source conflicts and turnout uncertainty keep the forecast deliberately uncertain.</p>
    <h2>Component Forecast</h2>
    <table><thead><tr><th>Component</th><th>Gallrein</th><th>Massie</th><th>Massie Margin</th><th>Weight</th></tr></thead><tbody>{component_rows}</tbody></table>
    <h2>Scenarios</h2>
    <table><thead><tr><th>Scenario</th><th>Gallrein</th><th>Massie</th><th>Massie Margin</th></tr></thead><tbody>{scenario_rows}</tbody></table>
    <h2>Market Comparison</h2>
    <table><thead><tr><th>Platform</th><th>Market Gallrein</th><th>Fair Gallrein</th><th>Gallrein Edge Pts</th></tr></thead><tbody>{market_rows}</tbody></table>
    <h2>Quality Gates</h2>
    <table><thead><tr><th>Signal</th><th>Status</th><th>Notes</th></tr></thead><tbody>{quality_rows}</tbody></table>
  </main>
</body>
</html>
"""
    (SHOWCASE_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"Showcase written to: {SHOWCASE_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
