from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_AS_OF = pd.Timestamp("2026-05-19")
AS_OF = DEFAULT_AS_OF
RNG_SEED = int(DEFAULT_AS_OF.strftime("%Y%m%d"))
VERSION = "1.0"


def configure_run(as_of: object | None = None) -> pd.Timestamp:
    global AS_OF, RNG_SEED
    AS_OF = pd.Timestamp(as_of).normalize() if as_of is not None else DEFAULT_AS_OF
    RNG_SEED = int(AS_OF.strftime("%Y%m%d"))
    return AS_OF


def read_raw() -> Dict[str, pd.DataFrame]:
    return {
        "polls": pd.read_csv(RAW_DIR / "polls.csv"),
        "poll_aggregators": pd.read_csv(RAW_DIR / "poll_aggregators.csv"),
        "markets": pd.read_csv(RAW_DIR / "markets.csv"),
        "market_timeseries": pd.read_csv(RAW_DIR / "market_timeseries.csv"),
        "finance_ads": pd.read_csv(RAW_DIR / "finance_ads.csv"),
        "endorsements_events": pd.read_csv(RAW_DIR / "endorsements_events.csv"),
        "turnout_signals": pd.read_csv(RAW_DIR / "turnout_signals.csv"),
        "turnout_history": pd.read_csv(RAW_DIR / "turnout_history.csv"),
        "district_baselines": pd.read_csv(RAW_DIR / "district_baselines.csv"),
        "pollster_ratings": pd.read_csv(RAW_DIR / "pollster_ratings.csv"),
        "source_checks": pd.read_csv(RAW_DIR / "source_checks.csv"),
        "source_conflicts": pd.read_csv(RAW_DIR / "source_conflicts.csv"),
        "source_registry": pd.read_csv(RAW_DIR / "source_registry.csv"),
        "wager_settings": pd.read_csv(RAW_DIR / "wager_settings.csv"),
    }


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_raw(raw: Dict[str, pd.DataFrame]) -> None:
    validate_sources(raw)
    require_columns(
        raw["polls"],
        [
            "race",
            "model_group",
            "pollster",
            "start_date",
            "end_date",
            "release_date",
            "sample_size",
            "population",
            "massie_pct",
            "gallrein_pct",
            "source_url",
            "notes",
        ],
        "polls",
    )
    core = raw["polls"][raw["polls"]["model_group"].eq("core")]
    if len(core) < 4:
        raise AssertionError("Expected at least four core late KY-04 polls.")
    if not core["massie_pct"].between(0, 100).all() or not core["gallrein_pct"].between(0, 100).all():
        raise AssertionError("Core poll percentages must be between 0 and 100.")
    require_columns(
        raw["pollster_ratings"],
        ["pollster", "pollster_reliability_weight", "source_url", "notes"],
        "pollster_ratings",
    )
    require_columns(
        raw["poll_aggregators"],
        [
            "aggregator",
            "record_type",
            "as_of_date",
            "massie_pct",
            "gallrein_pct",
            "include_in_model",
            "correlation_group",
            "source_quality",
            "source_url",
            "notes",
        ],
        "poll_aggregators",
    )
    active_aggregators = raw["poll_aggregators"][raw["poll_aggregators"]["include_in_model"].map(to_bool)]
    if active_aggregators["correlation_group"].duplicated().any():
        raise AssertionError("Active poll aggregator rows must have unique correlation_group values.")
    require_columns(
        raw["turnout_history"],
        ["signal_name", "signal_date", "metric", "value", "source_quality", "source_url", "notes"],
        "turnout_history",
    )
    require_columns(
        raw["district_baselines"],
        [
            "baseline_name",
            "signal_date",
            "metric",
            "value",
            "candidate_affinity",
            "source_quality",
            "source_url",
            "notes",
        ],
        "district_baselines",
    )
    require_columns(
        raw["source_conflicts"],
        ["conflict_id", "signal_date", "conflict_type", "affected_dataset", "severity", "source_quality", "source_url", "notes"],
        "source_conflicts",
    )

    registry_datasets = set(raw["source_registry"]["dataset_name"])
    expected = set(raw) - {"source_registry"}
    missing_registry = expected - registry_datasets
    if missing_registry:
        raise AssertionError(f"source_registry missing datasets: {sorted(missing_registry)}")


def validate_sources(raw: Dict[str, pd.DataFrame]) -> None:
    for name, frame in raw.items():
        if name in {"wager_settings", "source_registry"}:
            continue
        if "source_url" not in frame.columns:
            raise AssertionError(f"{name} is missing source_url")
        if "notes" not in frame.columns:
            raise AssertionError(f"{name} is missing notes")
        if frame["source_url"].fillna("").astype(str).str.strip().eq("").any():
            raise AssertionError(f"{name} has blank source_url values")
        if frame["notes"].fillna("").astype(str).str.strip().eq("").any():
            raise AssertionError(f"{name} has blank notes values")


def setting(raw: Dict[str, pd.DataFrame], key: str, default: float) -> float:
    settings = raw["wager_settings"].set_index("setting")["value"]
    if key not in settings:
        return default
    return float(settings[key])


def to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_polls(polls: pd.DataFrame) -> pd.DataFrame:
    out = polls.copy()
    for column in ["start_date", "end_date", "release_date"]:
        out[column] = pd.to_datetime(out[column], errors="coerce")
    out["sample_size"] = pd.to_numeric(out["sample_size"], errors="coerce").fillna(400)
    out["moe"] = pd.to_numeric(out["moe"], errors="coerce")
    out["massie_pct"] = pd.to_numeric(out["massie_pct"], errors="coerce")
    out["gallrein_pct"] = pd.to_numeric(out["gallrein_pct"], errors="coerce")
    out["undecided_pct"] = pd.to_numeric(out["undecided_pct"], errors="coerce").fillna(0)
    out["two_party_total"] = out["massie_pct"] + out["gallrein_pct"]
    out["massie_two_party_pct"] = 100 * out["massie_pct"] / out["two_party_total"]
    out["gallrein_two_party_pct"] = 100 * out["gallrein_pct"] / out["two_party_total"]
    out["massie_margin_points"] = out["massie_two_party_pct"] - out["gallrein_two_party_pct"]
    out["days_old"] = (AS_OF - out["end_date"]).dt.days.clip(lower=0)
    out["recency_weight"] = np.exp(-out["days_old"] / 10.0)
    out["sample_weight"] = np.sqrt(out["sample_size"] / 500.0)
    out["population_weight"] = out["population"].astype(str).str.contains("LV|likely", case=False, regex=True).map(
        {True: 1.0, False: 0.9}
    )
    out["mode_weight"] = np.where(out["mode"].astype(str).str.contains("unknown", case=False), 0.9, 1.0)
    out["partisan_weight"] = np.where(out["partisan_or_internal"].map(to_bool), 0.75, 1.0)
    ratings = pd.read_csv(RAW_DIR / "pollster_ratings.csv")
    ratings["pollster_reliability_weight"] = pd.to_numeric(
        ratings["pollster_reliability_weight"], errors="coerce"
    ).fillna(0.8)
    out = out.merge(
        ratings[["pollster", "pollster_reliability_weight", "pollster_rating_notes"]],
        on="pollster",
        how="left",
    )
    out["pollster_reliability_weight"] = out["pollster_reliability_weight"].fillna(0.8)
    out["source_verification_weight"] = np.select(
        [
            out["model_group"].eq("watchlist"),
            out["source_url"].astype(str).str.contains("predictionedge.com", case=False, regex=False),
        ],
        [0.65, 0.85],
        default=1.0,
    )

    core_mask = out["model_group"].eq("core")
    cluster_counts = out.loc[core_mask].groupby("pollster")["pollster"].transform("count")
    out.loc[core_mask, "cluster_weight"] = 1 / np.sqrt(cluster_counts)
    out["cluster_weight"] = out["cluster_weight"].fillna(1.0)
    out["poll_weight"] = (
        out["recency_weight"]
        * out["sample_weight"]
        * out["population_weight"]
        * out["mode_weight"]
        * out["partisan_weight"]
        * out["pollster_reliability_weight"]
        * out["source_verification_weight"]
        * out["cluster_weight"]
    )
    out["last_updated"] = str(AS_OF.date())
    return out


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.astype(float), weights=weights.astype(float)))


def weighted_std(values: pd.Series, weights: pd.Series) -> float:
    average = weighted_mean(values, weights)
    variance = np.average((values.astype(float) - average) ** 2, weights=weights.astype(float))
    return float(math.sqrt(max(variance, 0.0)))


def effective_sample_count(weights: pd.Series) -> float:
    weights = weights.astype(float)
    denom = float((weights**2).sum())
    if denom <= 0:
        return 0.0
    return float((weights.sum() ** 2) / denom)


def normal_probability(mean_margin: float, sigma: float) -> Dict[str, float]:
    sigma = max(float(sigma), 0.1)
    massie = 0.5 * (1 + math.erf(mean_margin / (sigma * math.sqrt(2))))
    massie = min(max(massie, 0.03), 0.97)
    return {
        "massie_win_probability": massie,
        "gallrein_win_probability": 1 - massie,
    }


def logit_probability_to_margin(probability: float, scale: float = 4.5) -> float:
    probability = min(max(float(probability), 0.03), 0.97)
    return float(np.clip(math.log(probability / (1 - probability)) * scale, -12.0, 12.0))


def conflict_sigma_penalty(conflicts: pd.DataFrame) -> float:
    severity_points = {"low": 0.15, "medium": 0.35, "high": 0.7}
    return float(conflicts["severity"].astype(str).str.lower().map(severity_points).fillna(0.25).sum())


def normalize_component_weights(components: pd.DataFrame) -> pd.DataFrame:
    out = components.copy()
    total = float(out["model_weight"].sum())
    if total <= 0:
        raise AssertionError("Component model weights must sum to a positive value.")
    out["model_weight"] = out["model_weight"] / total
    return out


def build_poll_diagnostics(polls: pd.DataFrame) -> pd.DataFrame:
    core = polls[polls["model_group"].eq("core")].copy()
    weighted_margin = weighted_mean(core["massie_margin_points"], core["poll_weight"])
    dispersion = weighted_std(core["massie_margin_points"], core["poll_weight"])
    eff_count = effective_sample_count(core["poll_weight"])
    latest = core["release_date"].max()
    rows = [
        {
            "diagnostic": "core_poll_count",
            "value": len(core),
            "notes": "Number of rows used in the core polling average.",
        },
        {
            "diagnostic": "effective_poll_count",
            "value": eff_count,
            "notes": "Cluster-adjusted effective count after same-pollster downweighting.",
        },
        {
            "diagnostic": "weighted_massie_margin_points",
            "value": weighted_margin,
            "notes": "Positive means Massie lead; negative means Gallrein lead.",
        },
        {
            "diagnostic": "weighted_pollster_dispersion_points",
            "value": dispersion,
            "notes": "Weighted standard deviation across core poll margins.",
        },
        {
            "diagnostic": "average_core_pollster_reliability_weight",
            "value": weighted_mean(core["pollster_reliability_weight"], core["poll_weight"]),
            "notes": "Weighted average explicit pollster reliability multiplier used in poll weights.",
        },
        {
            "diagnostic": "average_core_source_verification_weight",
            "value": weighted_mean(core["source_verification_weight"], core["poll_weight"]),
            "notes": "Weighted average row-level direct-source vs aggregator-source multiplier.",
        },
        {
            "diagnostic": "latest_core_poll_release_date",
            "value": str(latest.date()),
            "notes": "Latest public core poll release date in the snapshot.",
        },
    ]
    out = pd.DataFrame(rows)
    out["last_updated"] = str(AS_OF.date())
    return out


def build_adjustments(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    finance = raw["finance_ads"].copy()
    pro_g = finance.loc[finance["candidate_affinity"].eq("Gallrein"), "ad_support_m"].fillna(0).sum()
    pro_m = finance.loc[finance["candidate_affinity"].eq("Massie"), "ad_support_m"].fillna(0).sum()
    spending_margin = pro_g - pro_m
    spending_adjustment = float(np.clip(-0.08 * spending_margin, -0.9, 0.9))

    events = raw["endorsements_events"].copy()
    events["weight_points"] = pd.to_numeric(events["weight_points"], errors="coerce").fillna(0)
    event_adjustment = events.apply(
        lambda row: row["weight_points"] if row["candidate_affinity"] == "Massie" else -row["weight_points"],
        axis=1,
    ).sum()
    event_adjustment = float(np.clip(event_adjustment, -1.2, 1.2))

    rows = [
        {
            "component": "incumbency_local_network",
            "massie_margin_adjustment_points": 1.1,
            "notes": "Capped incumbency and local-network adjustment for a long-tenured district incumbent.",
        },
        {
            "component": "outside_spending",
            "massie_margin_adjustment_points": spending_adjustment,
            "notes": "Capped adjustment from pro-Gallrein vs pro-Massie outside spending; negative helps Gallrein.",
        },
        {
            "component": "endorsement_event_net",
            "massie_margin_adjustment_points": event_adjustment,
            "notes": "Capped net endorsement/event score; negative helps Gallrein.",
        },
    ]
    out = pd.DataFrame(rows)
    out["last_updated"] = str(AS_OF.date())
    return out


def build_scenarios(polls: pd.DataFrame, adjustments: pd.DataFrame) -> pd.DataFrame:
    core = polls[polls["model_group"].eq("core")].copy()
    polling_margin = weighted_mean(core["massie_margin_points"], core["poll_weight"])
    dispersion = weighted_std(core["massie_margin_points"], core["poll_weight"])
    eff_count = effective_sample_count(core["poll_weight"])
    adj = float(adjustments["massie_margin_adjustment_points"].sum())
    base_sigma = math.sqrt((5.0 / math.sqrt(max(eff_count, 1))) ** 2 + dispersion**2 + 8.0**2)
    base_sigma = float(np.clip(base_sigma, 8.5, 14.5))

    scenario_specs = [
        ("polling_only", polling_margin, base_sigma, "Core poll average only; no contextual adjustments."),
        (
            "full_model_mid_turnout",
            polling_margin + adj,
            base_sigma,
            "Core model with capped incumbency, spending, and endorsement/event adjustments.",
        ),
        (
            "older_high_propensity_turnout",
            polling_margin + adj - 2.0,
            base_sigma + 0.7,
            "High-propensity and older electorate stress case; based on late-poll age split.",
        ),
        (
            "younger_lower_propensity_turnout",
            polling_margin + adj + 2.0,
            base_sigma + 0.7,
            "Younger/lower-propensity electorate stress case; based on Big Data early-vote discussion.",
        ),
        (
            "massie_poll_error",
            polling_margin + adj + 5.0,
            base_sigma + 1.2,
            "Stress case where late polling overstates Gallrein or misses Massie local support.",
        ),
        (
            "gallrein_poll_error",
            polling_margin + adj - 5.0,
            base_sigma + 1.2,
            "Stress case where late polling understates Gallrein's Trump-aligned turnout.",
        ),
    ]
    rows = []
    for name, mean, sigma, notes in scenario_specs:
        prob = normal_probability(mean, sigma)
        rows.append(
            {
                "scenario": name,
                **prob,
                "mean_massie_margin_points": mean,
                "sigma_margin_points": sigma,
                "margin_80pct_low": mean - 1.28155 * sigma,
                "margin_80pct_high": mean + 1.28155 * sigma,
                "notes": notes,
            }
        )
    out = pd.DataFrame(rows)
    out["last_updated"] = str(AS_OF.date())
    return out


def build_market_timeseries(markets: pd.DataFrame) -> pd.DataFrame:
    out = markets.copy()
    out["mid"] = pd.to_numeric(out["mid"], errors="coerce")
    out["last"] = pd.to_numeric(out["last"], errors="coerce")
    out["market_prob"] = out["mid"].fillna(out["last"])
    major = out["candidate"].isin(["Thomas Massie", "Ed Gallrein"])
    major_total = out.loc[major, "market_prob"].sum()
    out["binary_normalized_prob"] = np.where(major, out["market_prob"] / major_total, np.nan)
    out["minor_market_artifact"] = ~major
    out["settlement_warning"] = out["minor_market_artifact"]
    out["liquidity_warning"] = pd.to_numeric(out["liquidity"], errors="coerce").fillna(0) < 5000
    out["last_updated"] = str(AS_OF.date())
    return out


def build_component_forecast(
    raw: Dict[str, pd.DataFrame],
    polls: pd.DataFrame,
    market_timeseries: pd.DataFrame,
    adjustments: pd.DataFrame,
) -> pd.DataFrame:
    core = polls[polls["model_group"].eq("core")].copy()
    polling_margin = weighted_mean(core["massie_margin_points"], core["poll_weight"])
    dispersion = weighted_std(core["massie_margin_points"], core["poll_weight"])
    eff_count = effective_sample_count(core["poll_weight"])
    poll_sigma = math.sqrt((5.0 / math.sqrt(max(eff_count, 1))) ** 2 + dispersion**2 + 8.0**2)
    poll_sigma = float(np.clip(poll_sigma, 8.5, 14.5))

    active_aggregators = raw["poll_aggregators"][raw["poll_aggregators"]["include_in_model"].map(to_bool)].copy()
    active_aggregators["massie_pct"] = pd.to_numeric(active_aggregators["massie_pct"], errors="coerce")
    active_aggregators["gallrein_pct"] = pd.to_numeric(active_aggregators["gallrein_pct"], errors="coerce")
    active_aggregators["two_party_total"] = active_aggregators["massie_pct"] + active_aggregators["gallrein_pct"]
    active_aggregators["massie_margin_points"] = (
        100 * active_aggregators["massie_pct"] / active_aggregators["two_party_total"]
        - 100 * active_aggregators["gallrein_pct"] / active_aggregators["two_party_total"]
    )
    aggregator_margin = weighted_mean(
        active_aggregators["massie_margin_points"],
        pd.Series(np.ones(len(active_aggregators)), index=active_aggregators.index),
    )
    aggregator_sigma = 9.8

    major_market = market_timeseries[market_timeseries["candidate"].isin(["Thomas Massie", "Ed Gallrein"])].copy()
    market_massie = float(
        major_market.loc[major_market["candidate"].eq("Thomas Massie"), "binary_normalized_prob"].iloc[0]
    )
    min_liquidity = pd.to_numeric(major_market["liquidity"], errors="coerce").fillna(0).min()
    market_margin = logit_probability_to_margin(market_massie)
    market_sigma = 8.5 + (0.8 if min_liquidity < 50000 else 0.0)

    baselines = raw["district_baselines"].copy()
    baselines["value"] = pd.to_numeric(baselines["value"], errors="coerce").fillna(0)
    fundamentals_margin = baselines.apply(
        lambda row: row["value"] if str(row["candidate_affinity"]) == "Massie" else -row["value"]
        if str(row["candidate_affinity"]) == "Gallrein"
        else 0,
        axis=1,
    ).sum()
    turnout = raw["turnout_history"].copy()
    turnout["value_numeric"] = pd.to_numeric(turnout["value"], errors="coerce")
    turnout_uncertainty = turnout.loc[turnout["metric"].eq("primary_turnout_uncertainty"), "value_numeric"].dropna()
    fundamentals_sigma = float(
        baselines.loc[baselines["metric"].eq("fundamentals_sigma"), "value"].dropna().iloc[0]
        if (baselines["metric"].eq("fundamentals_sigma")).any()
        else 10.5
    )
    if not turnout_uncertainty.empty:
        fundamentals_sigma += float(turnout_uncertainty.iloc[0]) / 2

    legacy_adjustment_margin = float(adjustments["massie_margin_adjustment_points"].sum())
    rows = [
        {
            "component": "direct_poll_model",
            "massie_margin_points": polling_margin,
            "sigma_margin_points": poll_sigma,
            "model_weight": 0.50,
            "evidence_count": len(core),
            "source_role": "highest_weight_direct_polling",
            "notes": "Verified late public polls with recency, sample, pollster, source, and cluster weights.",
        },
        {
            "component": "aggregator_anchor",
            "massie_margin_points": aggregator_margin,
            "sigma_margin_points": aggregator_sigma,
            "model_weight": 0.15,
            "evidence_count": len(active_aggregators),
            "source_role": "correlated_polling_anchor",
            "notes": "PSI average is used once as a correlated anchor; visible components are not double-counted.",
        },
        {
            "component": "bounded_market_model",
            "massie_margin_points": market_margin,
            "sigma_margin_points": market_sigma,
            "model_weight": 0.20,
            "evidence_count": len(major_market),
            "source_role": "bounded_market_signal",
            "notes": "Binary-normalized market probability transformed to a margin proxy and capped below polling weight.",
        },
        {
            "component": "fundamentals_turnout_model",
            "massie_margin_points": fundamentals_margin,
            "sigma_margin_points": fundamentals_sigma,
            "model_weight": 0.15,
            "evidence_count": len(baselines) + len(turnout),
            "source_role": "wide_prior_context",
            "notes": f"District, campaign, and turnout priors with wide uncertainty; legacy adjustment sum was {legacy_adjustment_margin:.3f} points.",
        },
    ]
    out = normalize_component_weights(pd.DataFrame(rows))
    for idx, row in out.iterrows():
        prob = normal_probability(row["massie_margin_points"], row["sigma_margin_points"])
        out.loc[idx, "massie_win_probability"] = prob["massie_win_probability"]
        out.loc[idx, "gallrein_win_probability"] = prob["gallrein_win_probability"]
    out["last_updated"] = str(AS_OF.date())
    return out


def build_ensemble_scenarios(components: pd.DataFrame, conflicts: pd.DataFrame) -> pd.DataFrame:
    final_margin = weighted_mean(components["massie_margin_points"], components["model_weight"])
    base_sigma = weighted_mean(components["sigma_margin_points"], components["model_weight"])
    component_dispersion = weighted_std(components["massie_margin_points"], components["model_weight"])
    sigma = float(np.clip(base_sigma + conflict_sigma_penalty(conflicts) + 0.25 * component_dispersion, 8.5, 14.5))

    scenario_specs = [
        (
            "polling_only",
            float(components.loc[components["component"].eq("direct_poll_model"), "massie_margin_points"].iloc[0]),
            float(components.loc[components["component"].eq("direct_poll_model"), "sigma_margin_points"].iloc[0]),
            "Direct late public poll model only.",
        ),
        (
            "aggregator_anchor",
            float(components.loc[components["component"].eq("aggregator_anchor"), "massie_margin_points"].iloc[0]),
            float(components.loc[components["component"].eq("aggregator_anchor"), "sigma_margin_points"].iloc[0]),
            "PSI correlated polling average used once, without component double-counting.",
        ),
        (
            "market_only",
            float(components.loc[components["component"].eq("bounded_market_model"), "massie_margin_points"].iloc[0]),
            float(components.loc[components["component"].eq("bounded_market_model"), "sigma_margin_points"].iloc[0]),
            "Prediction-market component only, after binary normalization and caps.",
        ),
        (
            "fundamentals_turnout",
            float(components.loc[components["component"].eq("fundamentals_turnout_model"), "massie_margin_points"].iloc[0]),
            float(components.loc[components["component"].eq("fundamentals_turnout_model"), "sigma_margin_points"].iloc[0]),
            "District, campaign, finance, endorsement, and turnout prior only.",
        ),
        (
            "full_model_mid_turnout",
            final_margin,
            sigma,
            "Bounded ensemble of direct polls, polling aggregator anchor, markets, and fundamentals.",
        ),
        (
            "older_high_propensity_turnout",
            final_margin - 2.0,
            sigma + 0.7,
            "High-propensity and older electorate stress case.",
        ),
        (
            "younger_lower_propensity_turnout",
            final_margin + 2.0,
            sigma + 0.7,
            "Younger/lower-propensity electorate stress case.",
        ),
        (
            "massie_poll_error",
            final_margin + 5.0,
            sigma + 1.2,
            "Stress case where late polling overstates Gallrein or misses Massie local support.",
        ),
        (
            "gallrein_poll_error",
            final_margin - 5.0,
            sigma + 1.2,
            "Stress case where late polling understates Gallrein's Trump-aligned turnout.",
        ),
    ]
    rows = []
    for name, mean, scenario_sigma, notes in scenario_specs:
        prob = normal_probability(mean, scenario_sigma)
        rows.append(
            {
                "scenario": name,
                **prob,
                "mean_massie_margin_points": mean,
                "sigma_margin_points": scenario_sigma,
                "margin_80pct_low": mean - 1.28155 * scenario_sigma,
                "margin_80pct_high": mean + 1.28155 * scenario_sigma,
                "notes": notes,
            }
        )
    out = pd.DataFrame(rows)
    out["last_updated"] = str(AS_OF.date())
    return out


def build_market_comparison(raw_markets: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    full = scenarios.loc[scenarios["scenario"].eq("full_model_mid_turnout")].iloc[0]
    rows = []
    for _, row in raw_markets.iterrows():
        massie_market = float(row["massie_implied_prob"])
        gallrein_market = float(row["gallrein_implied_prob"])
        total = massie_market + gallrein_market
        if total <= 0:
            continue
        massie_norm = massie_market / total
        gallrein_norm = gallrein_market / total
        rows.append(
            {
                "platform": row["platform"],
                "timestamp": row["timestamp"],
                "massie_market_probability": massie_norm,
                "gallrein_market_probability": gallrein_norm,
                "massie_fair_probability": float(full["massie_win_probability"]),
                "gallrein_fair_probability": float(full["gallrein_win_probability"]),
                "massie_edge_points": 100 * (float(full["massie_win_probability"]) - massie_norm),
                "gallrein_edge_points": 100 * (float(full["gallrein_win_probability"]) - gallrein_norm),
                "source_url": row["source_url"],
                "notes": "Market normalized to Massie/Gallrein binary universe for comparison only.",
            }
        )
    out = pd.DataFrame(rows)
    out["last_updated"] = str(AS_OF.date())
    return out


def build_wager_value_table(raw: Dict[str, pd.DataFrame], comparison: pd.DataFrame) -> pd.DataFrame:
    required_edge = setting(raw, "required_edge_points", 8.0)
    fee_buffer = setting(raw, "politics_fee_buffer_points", 1.0)
    threshold = required_edge + fee_buffer
    rows = []
    for _, row in comparison.iterrows():
        best_edge = max(float(row["massie_edge_points"]), float(row["gallrein_edge_points"]))
        if best_edge >= threshold:
            value_flag = "massie_value" if row["massie_edge_points"] >= row["gallrein_edge_points"] else "gallrein_value"
        else:
            value_flag = "no_bet_zone"
        rows.append(
            {
                "platform": row["platform"],
                "timestamp": row["timestamp"],
                "value_flag": value_flag,
                "decision_eligible": False,
                "required_edge_points": threshold,
                "massie_edge_points": row["massie_edge_points"],
                "gallrein_edge_points": row["gallrein_edge_points"],
                "block_reasons": "guarded_medium_reliability;pre_close_uncertainty;fair_odds_only_policy",
                "notes": "Fair-odds comparison only; no stake sizing or unconditional bet recommendation.",
            }
        )
    out = pd.DataFrame(rows)
    out["last_updated"] = str(AS_OF.date())
    return out


def build_sensitivity(polls: pd.DataFrame, adjustments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pollster in sorted(polls.loc[polls["model_group"].eq("core"), "pollster"].unique()):
        reduced = polls[~((polls["model_group"].eq("core")) & (polls["pollster"].eq(pollster)))].copy()
        if len(reduced[reduced["model_group"].eq("core")]) < 2:
            continue
        scenarios = build_scenarios(reduced, adjustments)
        full = scenarios.loc[scenarios["scenario"].eq("full_model_mid_turnout")].iloc[0]
        rows.append(
            {
                "sensitivity_case": f"drop_{pollster}",
                "massie_win_probability": full["massie_win_probability"],
                "gallrein_win_probability": full["gallrein_win_probability"],
                "mean_massie_margin_points": full["mean_massie_margin_points"],
                "notes": f"Core forecast excluding {pollster}.",
            }
        )
    out = pd.DataFrame(rows)
    out["last_updated"] = str(AS_OF.date())
    return out


def build_source_freshness(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    registry = raw["source_registry"].copy()
    rows = []
    for _, row in registry.iterrows():
        dataset = row["dataset_name"]
        frame = raw.get(dataset)
        latest = ""
        if frame is not None:
            date_cols = [c for c in frame.columns if "date" in c or c == "timestamp"]
            dates = []
            for col in date_cols:
                parsed = pd.to_datetime(frame[col], errors="coerce", utc=True, format="mixed")
                if parsed.notna().any():
                    dates.append(parsed.max().tz_convert(None))
            if dates:
                latest = max(dates)
        days_stale = None if latest == "" else int((AS_OF - pd.Timestamp(latest).tz_localize(None).normalize()).days)
        rows.append(
            {
                "dataset_name": dataset,
                "trust_tier": row["trust_tier"],
                "freshness_sla_days": row["freshness_sla_days"],
                "latest_source_date": "" if latest == "" else str(pd.Timestamp(latest).date()),
                "days_stale": days_stale,
                "freshness_pass": True if days_stale is None else days_stale <= int(row["freshness_sla_days"]),
                "source_url": row["citation_url"],
                "notes": row["notes"],
            }
        )
    out = pd.DataFrame(rows)
    out["last_updated"] = str(AS_OF.date())
    return out


def build_quality_report(raw: Dict[str, pd.DataFrame], polls: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    core = polls[polls["model_group"].eq("core")]
    full = scenarios.loc[scenarios["scenario"].eq("full_model_mid_turnout")].iloc[0]
    rows = [
        {
            "signal_group": "polls",
            "status": "usable",
            "metric": "core_poll_rows",
            "value": len(core),
            "notes": "Late public polls exist but disagree materially.",
        },
        {
            "signal_group": "official_results",
            "status": "excluded_pre_close",
            "metric": "pre_close_vote_rows",
            "value": 0,
            "notes": "Official live-results page was checked pre-close; no vote totals are used in forecast.",
        },
        {
            "signal_group": "market_prices",
            "status": "bounded_ensemble_component",
            "metric": "market_rows",
            "value": len(raw["markets"]),
            "notes": "Markets are included with caps, normalization, and settlement/liquidity checks.",
        },
        {
            "signal_group": "poll_aggregators",
            "status": "correlated_anchor",
            "metric": "active_aggregator_rows",
            "value": raw["poll_aggregators"]["include_in_model"].map(to_bool).sum(),
            "notes": "Aggregator averages are used once and visible components are retained as deduplication evidence.",
        },
        {
            "signal_group": "finance_ads",
            "status": "capped_context",
            "metric": "rows",
            "value": len(raw["finance_ads"]),
            "notes": "Extraordinary spending is capped because late polls already reflect much of it.",
        },
        {
            "signal_group": "forecast",
            "status": "guarded_medium_reliability",
            "metric": "massie_fair_probability",
            "value": full["massie_win_probability"],
            "notes": "Ensemble uses more signals, but conflict penalties keep uncertainty wide.",
        },
    ]
    out = pd.DataFrame(rows)
    out["last_updated"] = str(AS_OF.date())
    return out


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_source_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": file_hash(path),
                "bytes": path.stat().st_size,
                "last_updated": str(AS_OF.date()),
            }
        )
    return pd.DataFrame(rows)


def write_model_docs(scenarios: pd.DataFrame, diagnostics: pd.DataFrame, comparison: pd.DataFrame) -> None:
    full = scenarios.loc[scenarios["scenario"].eq("full_model_mid_turnout")].iloc[0]
    diagnostics_text = "\n".join(
        f"- {row.diagnostic}: {row.value} ({row.notes})" for row in diagnostics.itertuples()
    )
    comparison_text = "\n".join(
        f"- {row.platform} {row.timestamp}: market {row.gallrein_market_probability:.3f} vs fair {row.gallrein_fair_probability:.3f}"
        for row in comparison.itertuples()
    )
    model_card = f"""# KY-04 Republican Primary Model Card

As of {AS_OF.date()}, the core fair-odds estimate is Gallrein {full['gallrein_win_probability']:.1%} / Massie {full['massie_win_probability']:.1%}.

This is a pre-election forecast for a volatile single-district primary. It is not financial advice and does not claim certainty.

## Inputs

- Late public polls from Quantus, Big Data Poll, and GrayHouse.
- Public Sentiment Institute polling average as one correlated polling anchor.
- Turnout context from Kentucky election guidance and late poll subgroup notes.
- Campaign spending and endorsement/event context as capped adjustments.
- Prediction-market prices as a bounded ensemble component, with market tables still shown for fair-odds comparison.

## Limits

- Few independent pollsters.
- Strong turnout-model dependence.
- Large outside-spending and late-event environment.
- No official vote totals are used in the pre-close forecast.
"""
    audit_report = f"""# KY-04 Forecast Audit Report

Run date: {AS_OF.date()}

Headline fair odds:

- Ed Gallrein: {full['gallrein_win_probability']:.1%}
- Thomas Massie: {full['massie_win_probability']:.1%}
- Mean Massie margin: {full['mean_massie_margin_points']:.1f} points
- 80% margin interval: {full['margin_80pct_low']:.1f} to {full['margin_80pct_high']:.1f}

The forecast is a bounded pre-result ensemble. It uses markets as one capped component but still keeps wide uncertainty because the core data has few pollsters, high poll disagreement, and major turnout uncertainty.

Core polling diagnostics:

{diagnostics_text}

Market comparison:

{comparison_text}
"""
    (PROCESSED_DIR / "model_card.md").write_text(model_card, encoding="utf-8")
    (PROCESSED_DIR / "audit_report.md").write_text(audit_report, encoding="utf-8")


def write_outputs(outputs: Dict[str, pd.DataFrame], model_output: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    (PROCESSED_DIR / "model_output.json").write_text(json.dumps(model_output, indent=2), encoding="utf-8")


def main(as_of: object | None = None) -> Dict[str, pd.DataFrame]:
    configure_run(as_of)
    raw = read_raw()
    validate_raw(raw)

    polls = normalize_polls(raw["polls"])
    poll_diagnostics = build_poll_diagnostics(polls)
    adjustments = build_adjustments(raw)
    market_timeseries = build_market_timeseries(raw["market_timeseries"])
    component_forecast = build_component_forecast(raw, polls, market_timeseries, adjustments)
    scenarios = build_ensemble_scenarios(component_forecast, raw["source_conflicts"])
    market_comparison = build_market_comparison(raw["markets"], scenarios)
    wager_value = build_wager_value_table(raw, market_comparison)
    sensitivity = build_sensitivity(polls, adjustments)
    source_freshness = build_source_freshness(raw)
    quality = build_quality_report(raw, polls, scenarios)
    source_manifest = build_source_manifest()

    full = scenarios.loc[scenarios["scenario"].eq("full_model_mid_turnout")].iloc[0]
    component_weights = component_forecast.set_index("component")["model_weight"].to_dict()
    conflict_count = len(raw["source_conflicts"])
    reliability_score = max(
        0,
        min(
            100,
            58
            - 1.6 * float(full["sigma_margin_points"])
            - 1.2 * conflict_count
            + 2.0 * len(component_forecast)
            + 2.5 * len(polls[polls["model_group"].eq("core")]),
        ),
    )
    model_output = {
        "version": VERSION,
        "as_of": str(AS_OF.date()),
        "run_mode": "pre_election_no_results",
        "headline_forecast": {
            "massie_fair_probability": float(full["massie_win_probability"]),
            "gallrein_fair_probability": float(full["gallrein_win_probability"]),
            "mean_massie_margin_points": float(full["mean_massie_margin_points"]),
            "forecast_withheld": False,
        },
        "component_forecast": component_forecast.to_dict("records"),
        "ensemble_weights": {key: float(value) for key, value in component_weights.items()},
        "uncertainty": {
            "margin_80pct_interval": [float(full["margin_80pct_low"]), float(full["margin_80pct_high"])],
            "sigma_margin_points": float(full["sigma_margin_points"]),
            "reliability_class": "guarded_medium",
        },
        "model_health": {
            "reliability_score": float(reliability_score),
            "reliability_class": "guarded_medium",
            "official_results_used": False,
            "candidate_universe": ["Thomas Massie", "Ed Gallrein"],
            "source_conflict_count": conflict_count,
        },
        "confidence_diagnostics": {
            "component_count": int(len(component_forecast)),
            "market_component_weight": float(component_weights.get("bounded_market_model", 0.0)),
            "polling_component_weight": float(component_weights.get("direct_poll_model", 0.0)),
            "aggregator_component_weight": float(component_weights.get("aggregator_anchor", 0.0)),
            "fundamentals_component_weight": float(component_weights.get("fundamentals_turnout_model", 0.0)),
            "source_conflict_sigma_penalty": conflict_sigma_penalty(raw["source_conflicts"]),
            "official_results_policy": "excluded_pre_result_mode",
        },
        "source_conflicts": raw["source_conflicts"].to_dict("records"),
        "scenario_range": {
            "massie_low": float(scenarios["massie_win_probability"].min()),
            "massie_high": float(scenarios["massie_win_probability"].max()),
            "gallrein_low": float(scenarios["gallrein_win_probability"].min()),
            "gallrein_high": float(scenarios["gallrein_win_probability"].max()),
        },
        "betting_decision_summary": {
            "policy": "bounded_market_ensemble_with_no_bet_recommendation",
            "decision": "no_unconditional_bet_recommendation",
            "notes": "Markets are one capped forecast component; the wager table remains a conservative fair-odds comparison.",
        },
        "source_freshness": source_freshness.to_dict("records"),
    }

    outputs = {
        "polls": polls,
        "component_forecast": component_forecast,
        "model_scenarios": scenarios,
        "poll_diagnostics": poll_diagnostics,
        "adjustments": adjustments,
        "market_timeseries": market_timeseries,
        "market_comparison": market_comparison,
        "wager_value_table": wager_value,
        "sensitivity": sensitivity,
        "data_quality_report": quality,
        "source_registry_status": source_freshness,
        "source_manifest": source_manifest,
    }
    write_outputs(outputs, model_output)
    write_model_docs(scenarios, poll_diagnostics, market_comparison)
    print(
        "KY-04 forecast summary\n"
        f"As of: {model_output['as_of']}\n"
        f"Thomas Massie: {model_output['headline_forecast']['massie_fair_probability']:.1%}\n"
        f"Ed Gallrein: {model_output['headline_forecast']['gallrein_fair_probability']:.1%}\n"
        f"Mean Massie margin: {model_output['headline_forecast']['mean_massie_margin_points']:.1f}\n"
        f"Reliability: {model_output['model_health']['reliability_class']}\n"
        f"Files written to: {PROCESSED_DIR}\n"
    )
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=str(DEFAULT_AS_OF.date()))
    args = parser.parse_args()
    main(args.as_of)
