# %%
import numpy as np
import pandas as pd
import wandb
from datetime import datetime
from scipy import stats

api = wandb.Api()

# %%

# Project is specified by <entity/project-name>
runs = api.runs(
    "oxcsml/diffusion_manifold",
    filters={
        "createdAt": {"$gte": "2022-02-14T00:00:00.000Z"}
        # 'config.name': 'fire's
    },
)

summary_list, config_list, name_list = [], [], []

rows = []

for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    config = {"config/" + k: v for k, v in run.config.items() if not k.startswith("_")}

    # .name is the human-readable name of the run.
    name_list.append(run.name)

    rows.append(
        {
            "group": run.group,
            **run.summary._json_dict,
            **config,
        }
    )

runs_df = pd.DataFrame(rows)
runs_df["config/architecture/hidden_shapes"] = runs_df[
    "config/architecture/hidden_shapes"
].astype(str)
runs_df["config/scheduler/schedules"] = runs_df["config/scheduler/schedules"].astype(
    str
)
runs_df["config/scheduler/boundaries"] = runs_df["config/scheduler/boundaries"].astype(
    str
)
runs_df["config/splits"] = runs_df["config/splits"].astype(str)
config_cols = [k for k in runs_df.columns if k.startswith("config")]

# %%


def make_method(row):
    if "moser" in row["group"]:
        return "Moser Flow"

    elif "stereo" in row["group"]:
        return "Stereo SGM"
    elif "cnf" in row["group"]:
        return "CNF"
    else:
        return "RSGM"


runs_df["method"] = runs_df.apply(make_method, axis=1)
runs_df["dataset"] = runs_df["config/dataset/_target_"].replace(
    {
        "riemannian_score_sde.datasets.earth.Flood": "Flood",
        "riemannian_score_sde.datasets.earth.Earthquake": "Earthquake",
        "riemannian_score_sde.datasets.earth.Fire": "Fire",
        "riemannian_score_sde.datasets.earth.VolcanicErruption": "Volcano",
        "score_sde.datasets.vMFDataset": "vMF",
    }
)

# %%
pm_metric = "sem"
latex = False
metric = "val/logp"
bold = True


def make_table_from_metric(
    metric,
    results,
    val_metric=None,
    ci=0.95,
    latex=False,
    bold=True,
    drop_nans=False,
    show_group=False,
):
    if val_metric is None:
        val_metric = metric

    alpha = (1 - ci) / 2

    if drop_nans:
        results = results[results[metric].notna()]
        results = results[results[val_metric].notna()]

    def half_ci(group):
        data = group.to_numpy()
        sem = stats.sem(data)
        t2 = stats.t.ppf(1 - alpha, len(data) - 1) - stats.t.ppf(alpha, len(data) - 1)
        return sem * (t2 / 2)
        # return np.std(data)

    def lower_ci(group):
        data = group.to_numpy()
        sem = stats.sem(data)
        mean = data.mean()
        t = stats.t.ppf(alpha, len(data) - 1)
        return mean + sem * t

    def upper_ci(group):
        data = group.to_numpy()
        sem = stats.sem(data)
        mean = data.mean()
        t = stats.t.ppf(1 - alpha, len(data) - 1)
        return mean + sem * t

    def count(group):
        data = group.to_numpy()
        return np.prod(data.shape)

    results = (
        results.groupby(by=["group", "method", "dataset"])
        .agg(
            {
                metric: ["mean", "std", "sem", lower_ci, upper_ci, half_ci, count],
                val_metric: [
                    "mean",
                    "std",
                    "sem",
                    lower_ci,
                    upper_ci,
                    half_ci,
                    count,
                ],
            }
        )
        .reset_index()
    )

    group_max_idx = (
        results.groupby(by=["method", "dataset"]).transform(max)[val_metric]["mean"]
        == results[val_metric]["mean"]
    )
    table = results[group_max_idx]

    table = table[table["dataset"].isin(["Earthquake", "Fire", "Flood", "Volcano"])]

    if latex:

        def format_result(row):
            return (
                f"{{{-row[metric]['mean']:0.2f}_{{\pm {row[metric]['half_ci']:0.2f}}}}}"
            )

        def bold_result(row):
            return "\\bm" + row["result"] if row["bold"].any() else row["result"]

    else:

        def format_result(row):
            return f"{-row[metric]['mean']:0.2f} ± {row[metric]['half_ci']:0.2f}"

        def bold_result(row):
            return "* " + row["result"] if row["bold"].any() else row["result"]

    table["group_max"] = table.groupby(by=["dataset"]).transform(max)[metric]["mean"]
    table["group_max"] = table.apply(
        lambda row: table.index[table[metric]["mean"] == row["group_max"].squeeze()][0],
        axis=1,
    )
    table["bold"] = table.apply(
        lambda row: (
            table.loc[row["group_max"], (metric, "mean")].squeeze()
            < row[metric]["upper_ci"]
        )
        or (
            row[metric]["mean"]
            > table.loc[row["group_max"], (metric, "lower_ci")].squeeze()
        ),
        axis=1,
    )

    table["result"] = table.apply(format_result, axis=1)
    if bold:
        table["result"] = table.apply(bold_result, axis=1)

    if latex:
        table["result"] = table.apply(lambda row: "$" + row["result"] + "$", axis=1)

    table["count"] = table[(metric, "count")]

    cols = (
        ["method", "dataset", "group"] if show_group else ["method", "dataset", "count"]
    )

    table_flat = table[cols].pivot(index="method", columns="dataset")
    table_flat = table_flat.reindex(
        [
            "CNF",
            "Moser Flow",
            "Stereo SGM",
            "RSGM",
        ]
    )

    table_flat = table_flat.droplevel(level=0, axis=1)
    table_flat = table_flat.droplevel(level=0, axis=1)
    table_flat = table_flat[["Volcano", "Earthquake", "Flood", "Fire"]]
    table_flat.columns.name = None
    table_flat.index.name = None

    return table_flat


# %%
val_table = make_table_from_metric("val/logp", runs_df)
val_table
# %%
test_table = make_table_from_metric(
    "test/logp", runs_df, val_metric="val/logp", drop_nans=True
)
test_table

# %%
test_table = make_table_from_metric("test/logp", runs_df, val_metric="test/logp")
test_table
# %%
make_table_from_metric("val/logp", runs_df, show_group=True).to_csv("best_configs.csv")
# %%

print(
    make_table_from_metric(
        "test/logp", runs_df, val_metric="val/logp", drop_nans=True
    ).to_latex(escape=False)
)

# %%
