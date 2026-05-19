# KY-04 Forecast Audit Report

Run date: 2026-05-19

Headline fair odds:

- Ed Gallrein: 58.9%
- Thomas Massie: 41.1%
- Mean Massie margin: -2.6 points
- 80% margin interval: -17.4 to 12.2

The forecast is a bounded pre-result ensemble. It uses markets as one capped component but still keeps wide uncertainty because the core data has few pollsters, high poll disagreement, and major turnout uncertainty.

Core polling diagnostics:

- core_poll_count: 4 (Number of rows used in the core polling average.)
- effective_poll_count: 3.866989415627689 (Cluster-adjusted effective count after same-pollster downweighting.)
- weighted_massie_margin_points: -4.161815994116762 (Positive means Massie lead; negative means Gallrein lead.)
- weighted_pollster_dispersion_points: 4.136031150378266 (Weighted standard deviation across core poll margins.)
- average_core_pollster_reliability_weight: 0.8894361681720044 (Weighted average explicit pollster reliability multiplier used in poll weights.)
- average_core_source_verification_weight: 0.9673411202600805 (Weighted average row-level direct-source vs aggregator-source multiplier.)
- latest_core_poll_release_date: 2026-05-19 (Latest public core poll release date in the snapshot.)

Market comparison:

- Polymarket 2026-05-19T15:59:07-04:00: market 0.607 vs fair 0.589
- Kalshi 2026-05-13: market 0.480 vs fair 0.589
- Kalshi 2026-05-18: market 0.630 vs fair 0.589

Demographic turnout diagnostics:

- under_56: 36.5% of Quantus likely electorate; Massie 60.7, Gallrein 29.1
- 56_plus: 63.5% of Quantus likely electorate; Massie 31.7, Gallrein 59.1

Modeled endorsement signals:

- Gallrein: Donald Trump (formal_endorsement, national)
- Gallrein: Andy Barr (formal_endorsement, statewide_federal)
- Gallrein: Nate Morris (formal_endorsement, statewide_candidate)
- Gallrein: Pete Hegseth (late_surrogate_event, national)
- Massie: Rand Paul (formal_endorsement, statewide_federal)

Historical trend context:

- Massie's contested GOP primary share averaged 77.4% in 2020, 2022, and 2024; latest pre-2026 primary was 75.9% in 2024.
- Historical GOP primary support is strong but not rising; it looks like a stable incumbent floor rather than a cycle-by-cycle growth trend.
- General-election support has mostly lived in the low-to-upper 60s, with a 2016 high-water mark rather than a steady upward climb.

Turnout environment context:

- Official April 24, 2026 registration snapshot put Republicans at 311,007 of 600,319 registered voters in KY-04 (51.8%).
- Recent contested KY-04 Republican primaries turned out about 22.5% of registered Republicans on average; 2020 was 28.6%, 2022 was 21.8%, and 2024 was 16.9%.
- Big Data Poll had Massie at 57.6% among planned early no-excuse voters and Gallrein at 52.3% among planned Election Day voters.
- Quantus modeled the likely electorate as 63.5% age 56+ and 36.5% under 56. Older voters favored Gallrein 59.1-31.7, while younger voters favored Massie 60.7-29.1. That makes a higher-propensity older electorate Gallrein-favorable unless younger and early-vote participation beats the likely-voter mix.
