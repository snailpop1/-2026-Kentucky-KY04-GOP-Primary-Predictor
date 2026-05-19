# 2026 KY-04 Republican Primary Forecast

Reproducible forecast project for the May 19, 2026 Republican primary in Kentucky's 4th Congressional District between Thomas Massie and Ed Gallrein.

This project is modeled after the local Paxton/Cornyn forecast project, but it removes runoff-specific transfer logic and treats KY-04 as a binary pre-election forecast. It combines late public polling, a polling-aggregator anchor, turnout context, district fundamentals, campaign spending, endorsement/event signals, and prediction-market snapshots. Prediction markets are included only as a bounded ensemble component and are also shown as a fair-odds comparison.

This is election-forecasting research, not financial advice. The model does not guarantee outcomes and does not recommend a specific bet.

## Quick Start

```powershell
python -m pip install -r requirements.txt
python scripts/build_dataset.py --as-of 2026-05-19
python scripts/build_showcase.py
python -m pytest
```

Review these first:

- `data/processed/model_output.json`
- `data/processed/component_forecast.csv`
- `data/processed/model_card.md`
- `data/processed/audit_report.md`
- `data/processed/wager_value_table.csv`
- `showcase/index.html`

## Model Summary

- Polling baseline uses core late polls from Quantus, Big Data Poll, and GrayHouse.
- Public Sentiment Institute's KY-04 polling average is used once as a correlated polling anchor; visible component rows are retained for deduplication and conflict checks.
- Poll weights include recency, sample size, population, mode, partisan/internal status, explicit pollster reliability, direct-source verification, and same-pollster clustering.
- Repeated Big Data tracking rows are cluster-downweighted so one pollster does not dominate.
- Prediction markets are binary-normalized and included as a capped ensemble component, below the direct-polling weight.
- Turnout scenarios test older/high-propensity Gallrein-friendly and younger/lower-propensity Massie-friendly electorates.
- Spending and endorsement effects are capped because they are already partly reflected in late polling.
- Official vote totals are never used in `pre_election_no_results` mode.

## Source Baseline

- Kentucky official live results: https://vrsws.sos.ky.gov/liveresults/?autorefresh=true
- Kentucky voting calendar: https://www.kychamber.com/vote
- Quantus via LINK nky: https://linknky.com/news/2026/05/13/gallrein-leads-massie-quantas-poll-ky4-gop-primary/
- Big Data Poll: https://www.bigdatapoll.com/blog/initial-results-for-kentucky-house-district-4-republican-primary-tracking-poll/
- GrayHouse: https://www.grayhouse.com/post/kentucky4
- Public Sentiment Institute: https://www.publicsentimentinstitute.com/polling
- Polymarket API: https://gamma-api.polymarket.com/events/slug/ky-04-republican-primary-winner
- Spending summary: https://linknky.com/elections/2026/05/18/massie-gallrein-gop-primary-becomes-most-expensive-in-u-s-history/
