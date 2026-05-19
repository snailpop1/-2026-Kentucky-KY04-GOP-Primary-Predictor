# Pre-Election Refresh Checklist

1. Refresh `data/raw/polls.csv` only with public, source-linked polls.
2. Refresh `data/raw/poll_aggregators.csv` with source-linked polling averages and visible component rows; mark only deduplicated average anchors as model-active.
3. Refresh `data/raw/market_timeseries.csv` with pre-close bid, ask, last, volume, liquidity, and settlement notes.
4. Refresh `data/raw/turnout_signals.csv` and `data/raw/turnout_history.csv` with official early-vote or turnout reports, but do not include reported vote totals in a pre-election forecast.
5. Refresh `data/raw/district_baselines.csv` and `data/raw/source_conflicts.csv` for material fundamentals, source conflicts, and guardrail notes.
6. Record no-new-data source checks in `data/raw/source_checks.csv`.
7. Run `python scripts/build_dataset.py --as-of 2026-05-19`.
8. Run `python scripts/build_showcase.py`.
9. Run `python -m pytest`.
10. Review `model_output.json`, `component_forecast.csv`, `audit_report.md`, `model_card.md`, and `wager_value_table.csv`.

Do not treat the wager table as a standalone betting recommendation. Markets are a bounded forecast component, and the wager table remains a fair-odds comparison with a conservative no-bet buffer.
