from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_dataset import main
from scripts.build_showcase import main as build_showcase


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def test_raw_sources_are_auditable() -> None:
    for path in RAW_DIR.glob("*.csv"):
        if path.name in {"source_registry.csv", "wager_settings.csv"}:
            continue
        frame = pd.read_csv(path)
        assert "source_url" in frame.columns
        assert "notes" in frame.columns
        assert frame["source_url"].fillna("").astype(str).str.strip().ne("").all()
        assert frame["notes"].fillna("").astype(str).str.strip().ne("").all()


def test_poll_schema_and_core_inventory() -> None:
    polls = pd.read_csv(RAW_DIR / "polls.csv")
    required = {
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
    }
    assert required.issubset(polls.columns)
    core = polls[polls["model_group"].eq("core")]
    assert len(core) >= 4
    assert {"Quantus Insights", "Big Data Poll", "GrayHouse"}.issubset(set(core["pollster"]))

    ratings = pd.read_csv(RAW_DIR / "pollster_ratings.csv")
    assert set(core["pollster"]).issubset(set(ratings["pollster"]))
    assert ratings["pollster_reliability_weight"].between(0.5, 1.05).all()


def test_source_registry_covers_raw_inputs() -> None:
    registry = pd.read_csv(RAW_DIR / "source_registry.csv")
    datasets = set(registry["dataset_name"])
    raw = {path.stem for path in RAW_DIR.glob("*.csv")} - {"source_registry"}
    assert raw.issubset(datasets)


def test_new_raw_signal_schemas_are_auditable() -> None:
    required_by_file = {
        "poll_aggregators.csv": {
            "aggregator",
            "record_type",
            "include_in_model",
            "correlation_group",
            "source_quality",
            "source_url",
            "notes",
        },
        "turnout_history.csv": {"signal_name", "signal_date", "metric", "value", "source_quality", "source_url", "notes"},
        "district_baselines.csv": {
            "baseline_name",
            "signal_date",
            "metric",
            "value",
            "candidate_affinity",
            "source_quality",
            "source_url",
            "notes",
        },
        "source_conflicts.csv": {
            "conflict_id",
            "signal_date",
            "conflict_type",
            "severity",
            "source_quality",
            "source_url",
            "notes",
        },
    }
    for filename, required in required_by_file.items():
        frame = pd.read_csv(RAW_DIR / filename)
        assert required.issubset(frame.columns)
        assert frame["source_quality"].fillna("").astype(str).str.strip().ne("").all()
        assert frame["source_url"].fillna("").astype(str).str.strip().ne("").all()
        assert frame["notes"].fillna("").astype(str).str.strip().ne("").all()


def test_poll_aggregator_average_is_not_double_counted() -> None:
    aggregators = pd.read_csv(RAW_DIR / "poll_aggregators.csv")
    active = aggregators[aggregators["include_in_model"].astype(str).str.lower().eq("true")]
    assert len(active) == 1
    assert active["record_type"].eq("average").all()
    assert active["correlation_group"].is_unique
    assert aggregators.loc[aggregators["record_type"].eq("component_visible"), "include_in_model"].astype(str).str.lower().eq(
        "false"
    ).all()


def test_model_outputs_are_valid_and_pre_results() -> None:
    main("2026-05-19")
    output = json.loads((PROCESSED_DIR / "model_output.json").read_text(encoding="utf-8"))
    headline = output["headline_forecast"]
    assert output["run_mode"] == "pre_election_no_results"
    assert output["model_health"]["official_results_used"] is False
    assert output["model_health"]["candidate_universe"] == ["Thomas Massie", "Ed Gallrein"]
    assert output["model_health"]["reliability_class"] == "guarded_medium"
    assert output["confidence_diagnostics"]["official_results_policy"] == "excluded_pre_result_mode"
    assert output["ensemble_weights"]["bounded_market_model"] <= 0.25
    assert 0.03 <= headline["massie_fair_probability"] <= 0.97
    assert 0.03 <= headline["gallrein_fair_probability"] <= 0.97
    assert abs(headline["massie_fair_probability"] + headline["gallrein_fair_probability"] - 1) < 1e-9


def test_processed_outputs_exist_and_have_last_updated() -> None:
    main("2026-05-19")
    required = [
        "polls.csv",
        "component_forecast.csv",
        "model_scenarios.csv",
        "poll_diagnostics.csv",
        "adjustments.csv",
        "market_timeseries.csv",
        "market_comparison.csv",
        "wager_value_table.csv",
        "sensitivity.csv",
        "data_quality_report.csv",
        "source_registry_status.csv",
        "source_manifest.csv",
        "model_output.json",
        "model_card.md",
        "audit_report.md",
    ]
    for filename in required:
        path = PROCESSED_DIR / filename
        assert path.exists(), f"Missing {filename}"
        assert path.stat().st_size > 0, f"Empty {filename}"
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            assert "last_updated" in frame.columns
            assert set(frame["last_updated"].astype(str)) == {"2026-05-19"}


def test_market_normalization_and_no_unconditional_bet() -> None:
    main("2026-05-19")
    markets = pd.read_csv(PROCESSED_DIR / "market_timeseries.csv")
    major = markets[markets["candidate"].isin(["Thomas Massie", "Ed Gallrein"])]
    assert abs(major["binary_normalized_prob"].sum() - 1) < 1e-9
    assert markets.loc[~markets["candidate"].isin(["Thomas Massie", "Ed Gallrein"]), "minor_market_artifact"].all()

    value = pd.read_csv(PROCESSED_DIR / "wager_value_table.csv")
    assert (value["decision_eligible"] == False).all()
    assert set(value["value_flag"]).issubset({"no_bet_zone", "massie_value", "gallrein_value"})
    assert value["block_reasons"].str.contains("fair_odds_only_policy").all()

    components = pd.read_csv(PROCESSED_DIR / "component_forecast.csv")
    assert abs(components["model_weight"].sum() - 1) < 1e-9
    assert components.loc[components["component"].eq("bounded_market_model"), "model_weight"].iloc[0] <= 0.25


def test_pre_result_mode_excludes_official_vote_totals() -> None:
    main("2026-05-19")
    output = json.loads((PROCESSED_DIR / "model_output.json").read_text(encoding="utf-8"))
    quality = pd.read_csv(PROCESSED_DIR / "data_quality_report.csv")
    conflicts = pd.read_csv(RAW_DIR / "source_conflicts.csv")
    assert output["run_mode"] == "pre_election_no_results"
    assert output["model_health"]["official_results_used"] is False
    assert quality.loc[quality["signal_group"].eq("official_results"), "status"].iloc[0] == "excluded_pre_close"
    assert conflicts["conflict_type"].str.contains("pre_result_guardrail").any()


def test_source_conflicts_widen_ensemble_uncertainty() -> None:
    main("2026-05-19")
    components = pd.read_csv(PROCESSED_DIR / "component_forecast.csv")
    scenarios = pd.read_csv(PROCESSED_DIR / "model_scenarios.csv")
    weighted_component_sigma = float((components["sigma_margin_points"] * components["model_weight"]).sum())
    full_sigma = float(
        scenarios.loc[scenarios["scenario"].eq("full_model_mid_turnout"), "sigma_margin_points"].iloc[0]
    )
    assert full_sigma > weighted_component_sigma


def test_poll_weights_include_recency_and_reliability() -> None:
    main("2026-05-19")
    polls = pd.read_csv(PROCESSED_DIR / "polls.csv")
    required_weights = {
        "recency_weight",
        "sample_weight",
        "pollster_reliability_weight",
        "source_verification_weight",
        "cluster_weight",
        "poll_weight",
    }
    assert required_weights.issubset(polls.columns)
    assert polls["pollster_reliability_weight"].between(0.5, 1.05).all()
    assert polls["source_verification_weight"].between(0.5, 1.0).all()


def test_showcase_builds() -> None:
    main("2026-05-19")
    build_showcase()
    path = ROOT / "showcase" / "index.html"
    assert path.exists()
    assert "KY-04 Republican Primary Forecast" in path.read_text(encoding="utf-8")
