# %%
import pandas as pd
import wandb

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("oxcsml/diffusion_manifold")

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
    raw_results,
    val_metric=None,
    pm_metric="sem",
    latex=False,
    bold=True,
    show_group=False,
):
    if val_metric is None:
        val_metric = metric

    results = (
        runs_df.groupby(by=["group", "method", "dataset"])
        .agg(
            {
                metric: ["mean", pm_metric],
                val_metric: ["mean", "std", "sem"],
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
                f"{{{-row[metric]['mean']:0.2f}_{{\pm {row[metric][pm_metric]:0.2f}}}}}"
            )

        def bold_result(row):
            return "\\bm" + row["result"] if row["bold"].any() else row["result"]

    else:

        def format_result(row):
            return f"{-row[metric]['mean']:0.2f} ± {row[metric][pm_metric]:0.2f}"

        def bold_result(row):
            return "* " + row["result"] if row["bold"].any() else row["result"]

    table["bold"] = (
        table.groupby(by=["dataset"]).transform(max)[metric]["mean"]
        == table[metric]["mean"]
    )

    table["result"] = table.apply(format_result, axis=1)
    if bold:
        table["result"] = table.apply(bold_result, axis=1)

    if latex:
        table["result"] = table.apply(lambda row: "$" + row["result"] + "$", axis=1)

    cols = (
        ["method", "dataset", "group"]
        if show_group
        else ["method", "dataset", "result"]
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
test_table = make_table_from_metric("test/logp", runs_df, val_metric="val/logp")
test_table
# %%
test_table = make_table_from_metric("test/logp", runs_df, val_metric="test/logp")
test_table
# %%
make_table_from_metric("val/logp", runs_df, show_group=True).to_csv("best_configs.csv")
# %%

print(
    make_table_from_metric(
        "val/logp", results, pm_metric="sem", latex=True, bold=True
    ).to_latex(escape=False)
)

# %%
