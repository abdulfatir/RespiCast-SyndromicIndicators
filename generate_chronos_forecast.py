from pathlib import Path

import pandas as pd
from chronos import Chronos2Pipeline

RESPICAST_ROOT = Path(__file__).parent
TARGET_DATA_PATH = RESPICAST_ROOT / ("target-data")
SUPPORTING_FILES_PATH = RESPICAST_ROOT / ("supporting-files")
MODEL_OUTPUT_PATH = RESPICAST_ROOT / ("model-output")
EVAL_PATH = RESPICAST_ROOT / ("model-evaluation")
REQUIRED_QUANTILES = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    0.99,
]


def load_snapshot_data(snapshot_path: Path) -> pd.DataFrame:
    """Load and parse a snapshot CSV file."""
    df = pd.read_csv(snapshot_path)
    df["truth_date"] = pd.to_datetime(df["truth_date"])
    return df


def load_forecasting_weeks(origin_date: str) -> pd.DataFrame:
    """Load the horizon to target_end_date mapping for a given origin_date."""
    df = pd.read_csv(SUPPORTING_FILES_PATH / "forecasting_weeks.csv")
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    df["target_end_date"] = pd.to_datetime(df["target_end_date"])
    return df[df["origin_date"] == origin_date]


def prepare_dataframe(raw_df: pd.DataFrame, last_observation_date: pd.Timestamp):
    try:
        return (
            raw_df.set_index("truth_date")
            .groupby("location")
            .resample("W-SUN")["value"]  # resample to weekly, there are many gaps
            .asfreq()
            .reset_index()
            .set_index("truth_date")
            .groupby("location")["value"]
            .apply(  # ensure that the last timestamp is one week before the forecast start
                lambda g: g.reindex(
                    pd.date_range(
                        start=g.index.min(),
                        end=last_observation_date,
                        freq="W-SUN",
                        name="truth_date",
                    ),
                )
            )
            .reset_index()
        )
    except Exception as ex:
        return pd.DataFrame()


def load_and_prepare_context(target_type: str, last_observation_date: pd.Timestamp) -> pd.DataFrame:

    erviss_snapshot = TARGET_DATA_PATH / "ERVISS" / f"latest-{target_type}_incidence.csv"
    erviss_df = load_snapshot_data(erviss_snapshot)
    fluid_snapshot = TARGET_DATA_PATH / "FluID" / f"latest-{target_type}_incidence.csv"
    fluid_df = load_snapshot_data(fluid_snapshot)

    context_df = pd.concat(
        [
            prepare_dataframe(erviss_df, last_observation_date=last_observation_date),
            prepare_dataframe(fluid_df, last_observation_date=last_observation_date),
        ],
        ignore_index=True,
    )

    return context_df


def validate_and_format_predictions(
    pred_df: pd.DataFrame, forecast_weeks: pd.DataFrame, origin_date: str, target_type: str
):
    # Load expected future dates, removing backcast and nowcast dates
    forecast_weeks = forecast_weeks.query("`horizon` > 0").reset_index(drop=True)
    expected_dates = pd.to_datetime(forecast_weeks["target_end_date"])
    formatted_pred_df = pred_df.copy().drop(columns=["target_name"]).rename(columns={"truth_date": "target_end_date"})
    assert (
        formatted_pred_df.groupby("location")
        .apply(lambda g: g["target_end_date"].reset_index(drop=True).equals(expected_dates), include_groups=False)
        .all()
    )

    formatted_pred_df = formatted_pred_df.melt(id_vars=["location", "target_end_date"], var_name="output_type_id")
    output_type_ids = formatted_pred_df["output_type_id"]
    formatted_pred_df["output_type"] = output_type_ids.map(lambda x: "median" if x == "predictions" else "quantile")
    formatted_pred_df["output_type_id"] = output_type_ids.map(lambda x: "" if x == "predictions" else x)
    formatted_pred_df = formatted_pred_df.merge(forecast_weeks[["target_end_date", "horizon"]], on="target_end_date")

    formatted_pred_df["origin_date"] = origin_date
    formatted_pred_df["target"] = f"{target_type} incidence"

    formatted_pred_df["value"] = formatted_pred_df["value"].clip(0)

    return formatted_pred_df[
        ["origin_date", "target", "target_end_date", "horizon", "location", "output_type", "output_type_id", "value"]
    ]


def forecast_chronos2_for_origin_date(
    pipeline: Chronos2Pipeline, origin_date: str = "2026-01-14", target_type="ILI", cross_learning: bool = True
):

    forecast_weeks = load_forecasting_weeks(origin_date)
    assert len(forecast_weeks) == 6
    last_observation_date = forecast_weeks.query("`horizon` == 0")["target_end_date"].iloc[0]

    context_df = load_and_prepare_context(target_type, last_observation_date)
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=4,
        quantile_levels=REQUIRED_QUANTILES,
        id_column="location",
        timestamp_column="truth_date",
        target="value",
        validate_inputs=False,
        cross_learning=cross_learning,
    )
    formatted_pred_df = validate_and_format_predictions(pred_df, forecast_weeks, origin_date, target_type)

    return formatted_pred_df


def main(model_id: str = "amazon/chronos-2", cross_learning: bool = True, device: str = "cpu"):
    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device)
    latest_origin_date = sorted(pd.read_csv(SUPPORTING_FILES_PATH / "forecasting_weeks.csv")["origin_date"].unique())[
        -1
    ]
    preds = pd.concat(
        [
            forecast_chronos2_for_origin_date(
                pipeline, latest_origin_date, target_type="ARI", cross_learning=cross_learning
            ),
            forecast_chronos2_for_origin_date(
                pipeline, latest_origin_date, target_type="ILI", cross_learning=cross_learning
            ),
        ],
        ignore_index=True,
    )
    team_model = "Chronos-Chronos2" + ("" if cross_learning else "uni")
    preds_csv_path = MODEL_OUTPUT_PATH / f"{team_model}/{latest_origin_date}-{team_model}.csv"
    preds_csv_path.parent.mkdir(exist_ok=True)
    preds.to_csv(preds_csv_path, index=False)


if __name__ == "__main__":
    main()
