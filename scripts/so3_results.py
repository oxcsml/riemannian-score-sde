# %%
import os
from datetime import datetime
from functools import partial

import wandb

import numpy as np
import pandas as pd
from scipy import stats

from IPython import get_ipython
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import seaborn as sns

api = wandb.Api()

# cmap_name = "plasma_r"
cmap_name = "viridis_r"

# helpers


def query(data_frame, query_string):
    if query_string == "all":
        return data_frame
    return data_frame.query(query_string)


ci = 0.95
alpha = (1 - ci) / 2


def half_ci(group):
    data = group.dropna().to_numpy()
    sem = stats.sem(data)
    t2 = stats.t.ppf(1 - alpha, len(data) - 1) - stats.t.ppf(alpha, len(data) - 1)
    return sem * (t2 / 2)


def lower_ci(group):
    data = group.dropna().to_numpy()
    sem = stats.sem(data)
    mean = data.mean()
    t = stats.t.ppf(alpha, len(data) - 1)
    return mean + sem * t


def upper_ci(group):
    data = group.dropna().to_numpy()
    sem = stats.sem(data)
    mean = data.mean()
    t = stats.t.ppf(1 - alpha, len(data) - 1)
    return mean + sem * t


def select_hyperparms(results, group_by, metric, val_metric, drop_nans=False, best=max):
    if drop_nans:
        results = results[results[metric].notna()]
        results = results[results[val_metric].notna()]

    reducers = ["mean", "std", "sem", lower_ci, upper_ci, half_ci, "count"]
    metric = metric if isinstance(metric, list) else [metric]
    metrics_to_agg = {key: reducers for key in metric + [val_metric]}
    results = results.groupby(by=["group"] + group_by).agg(metrics_to_agg)
    # .reset_index()

    group_max_idx = (
        results.groupby(by=group_by).transform(best)[val_metric]["mean"]
        == results[val_metric]["mean"]
    )
    data = results[group_max_idx]

    return data, group_max_idx


#%%
# Project is specified by <entity/project-name>
runs = api.runs(
    "oxcsml/diffusion_manifold",
    filters={
        "createdAt": {"$gte": "2022-05-01T00:00:00.000Z"},
        # 'config.name': 'fire's
        "username": {"$eq": "emilem"},
    },
)

summary_list, config_list, name_list = [], [], []

rows = []

for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    config = {k: v for k, v in run.config.items() if not k.startswith("_")}

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
# format to str
fields = []
for field in fields:
    runs_df[field] = runs_df[field].astype(str)
# config_cols = [k for k in runs_df.columns if k.startswith("config")]

# format data
def make_method(row):
    if "moser" in row["group"]:
        return "Moser Flow"
    elif "sgm_exp" in row["group"]:
        return "Exp-wrapped SGM"
    elif "cnf" in row["group"]:
        return "CNF"
    else:
        return "RSGM"


runs_df["method"] = runs_df.apply(make_method, axis=1)
runs_df["dataset"] = runs_df["dataset/K"]  # .astype(int)
runs_df["loss"] = runs_df["loss/_target_"].replace(
    {
        "riemannian_score_sde.losses.get_ism_loss_fn": "ism",
        "riemannian_score_sde.losses.get_dsm_loss_fn": "dsm",
        "riemannian_score_sde.losses.get_moser_loss_fn": "moser",
    }
)
# runs_df["dataset"] = runs_df["config/dataset/_target_"].replace(
#     {
#         "riemannian_score_sde.datasets.earth.Flood": "Flood",
#         "riemannian_score_sde.datasets.earth.Earthquake": "Earthquake",
#         "riemannian_score_sde.datasets.earth.Fire": "Fire",
#         "riemannian_score_sde.datasets.earth.VolcanicErruption": "Volcano",
#         "score_sde.datasets.vMFDataset": "vMF",
#     }
# )

#%%
# Plotting


def generate_plot(data, metric, x_metric):
    get_ipython().run_line_magic("matplotlib", "inline")
    colours = ["C{}".format(i) for i in range(10)] + ["C{}".format(i) for i in range(10)]
    linestyles = ["solid", (0, (5, 5)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    fontsize = 20

    fig, axis = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(10, 6))

    i = 0
    for j, (index, new_df) in enumerate(data.groupby(level=0)):
        # print(i)
        # print(index)
        x = new_df[(x_metric)]
        y, y_low, y_up, y_err, sem, count = (
            np.array(new_df[(metric, "mean")]),
            np.array(new_df[(metric, "lower_ci")].fillna(0)),
            np.array(new_df[(metric, "upper_ci")].fillna(0)),
            np.array(new_df[(metric, "half_ci")].fillna(0)),
            np.array(new_df[(metric, "sem")].fillna(0)),
            np.array(new_df[(metric, "count")]),
        )
        yerr = np.stack([y - y_up, y_low - y], axis=0)
        # yerr = sem
        label = f"{new_df.index[0]}"
        # axis.plot(x, y, lw=2, color=colours[j], linestyle=linestyles[i], label=label)
        # axis.fill_between(x, y_low, y_up, alpha=0.2, facecolor=colours[j])
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            lw=2,
            color=colours[j],
            linestyle=linestyles[i],
            label=label,
            elinewidth=6,
            alpha=0.3,
        )

    # axis.set_xscale("log", base=2)
    axis.set_xlabel(x_metric, fontsize=fontsize)
    axis.set_ylabel(metric, fontsize=fontsize)
    axis.legend(fontsize=0.8 * fontsize)
    axis.grid(True)

    return fig, axis

    # %%


# ************************************************************ #
# ********************* logp vs nb mixture ******************* #
# ************************************************************ #

criteria = [
    "`name` == 'SO3'",
    "`dataset/conditional` == False",
    "`steps` == 100000",
    "`warmup_steps` == 100",
    # "`loss/lambda_w` == null",
    # "`optim/learning_rate` in [0.0002, 0.0005]",
    # "`optim/learning_rate` in [0.0005]",
    # "`dataset` in [8,32,128,512]",
    "`dataset` in [16,32,64]",
    # "`dataset` in [8,16,32,64,128,256,512]",
    # "(method == 'Moser Flow' | (method != 'Moser Flow' & `flow/beta_0` == 0.001))",
    "(method == 'Moser Flow' | (method != 'Moser Flow' & `flow/beta_f` in [0.5,1, 2, 4, 6, 8, 10, 12]))",
    # "(method == 'Moser Flow' | (method != 'Moser Flow' & `optim/learning_rate` == 0.0002))",
    # "(method != 'Moser Flow' | (method == 'Moser Flow' & `optim/learning_rate` == 0.0002))",
    # "(method != 'Moser Flow' | (method == 'Moser Flow' & `loss/K` == 10000))",
    # "(method != 'Moser Flow' | (method == 'Moser Flow' & `loss/alpha_m` == 1))",
    # "`loss` == 'ism'",
    # "`loss` == 'dsm'",
    # "(method != 'Moser' & `loss` == 'dsm') | method == 'Moser'",
    "(method != 'Moser Flow' | (method == 'Moser Flow' & `pushf/diffeq` == True))",
    "(method != 'RSGM' | (method == 'RSGM' & `flow/N` == 100))",
]
criteria = ["all"] if criteria == [] else criteria
results = query(runs_df, " & ".join(criteria))
results = results.rename(columns={"test/logp": "log-like", "test/nfe": "NFE"})
results.loc[:, "NFE"] = results.loc[:, "NFE"] / 1000
# results.loc[:, "log-like"] = -results.loc[:, "log-like"]

x_metric = "dataset"
metrics = ["log-like", "NFE"]
# metrics = ["test/logp"]
# metrics = "test/logp"
val_metric = "log-like"
# val_metric = "val/logp"
group_by = [x_metric, "method"]

data, _ = select_hyperparms(results, group_by, metrics, val_metric)
data = data.sort_values(by=[x_metric])
# data = data.reindex(["Moser Flow", "Exp-wrapped SGM", "RSGM"], axis=0, level=2)
# data
#%%


def format_result(metric, fmt, row):
    mean, err = row[metric]["mean"], row[metric]["half_ci"]
    mean = fmt.format(mean)
    err = fmt.format(err)
    return f"{{{mean}_{{\pm {err}}}}}"
    # return f"{{{mean:0.2f}_{{\pm {err:0.2f}}}}}"


def bold_result(key, row):
    return "\\bm" + row[key] if row["bold"].any() else row[key]


table = data.reset_index(level=0)
# table = table.sort_values(by=[x_metric])
# table = table.sort_index()

# data = data[[(metric, "mean"), (metric, "sem")]]

for metric, best, fmt in zip(metrics, [max, min], ["{:.2f}", "{:.1f}"]):
    table["group_best"] = table.groupby(by=["dataset"]).transform(best)[metric]["mean"]
    table["group_best"] = table.apply(
        lambda row: table.index[table[metric]["mean"] == row["group_best"].squeeze()][0],
        axis=1,
    )
    if best == max:
        table["bold"] = table.apply(
            lambda row: (
                table.loc[row["group_best"], (metric, "mean")].squeeze()
                < row[metric]["upper_ci"]
            )
            or (
                row[metric]["mean"]
                > table.loc[row["group_best"], (metric, "lower_ci")].squeeze()
            ),
            axis=1,
        )
    else:
        table["bold"] = table.apply(
            lambda row: (
                table.loc[row["group_best"], (metric, "mean")].squeeze()
                > row[metric]["lower_ci"]
            )
            or (
                row[metric]["mean"]
                < table.loc[row["group_best"], (metric, "upper_ci")].squeeze()
            ),
            axis=1,
        )

    key = f"result_{metric}"
    table[key] = table.apply(partial(format_result, metric, fmt), axis=1)
    table[key] = table.apply(partial(bold_result, key), axis=1)
    table[key] = table.apply(lambda row: "$" + row[key] + "$", axis=1)

table = table.drop(columns=["group", "group_best", "bold"] + metrics)
for metric in metrics:
    key = f"result_{metric}"
    table[metric] = table[key]
    table = table.drop(columns=[key])

table.index = table.index.set_levels(table.index.levels[0].astype(int), level=0)
table.index = table.index.set_names({"dataset": "M", "method": "Method"})

table = table.droplevel(1, axis=1)
table = table.unstack(level=0)
table = table.swaplevel(i=0, j=1, axis=1)
table = table.reindex(sorted(table.columns), axis=1)
table = table.reindex(metrics, level=1, axis=1)
table = table.reindex(["Moser Flow", "Exp-wrapped SGM", "RSGM"], axis=0)
for method in ["Exp-wrapped SGM", "RSGM"]:
    table.loc[[method]] = table.loc[[method]].applymap(
        lambda x: "\cellcolor{pearDark!20} " + x
    )

# cols = ["method", "dataset", "result"]

# table_flat = table_flat.droplevel(level=0, axis=1)
# table_flat = table_flat.droplevel(level=0, axis=1)
# table_flat = table_flat[["Volcano", "Earthquake", "Flood", "Fire"]]
# table_flat.columns.name = None
# table_flat.index.name = None


latex_path = "../doc/tables/so3.tex"
filename = os.path.join(os.getcwd(), latex_path)
table.style.to_latex(
    buf=filename, hrules=True, multicol_align="c", column_format="lrrrrrrrr"
)  # , column_format="cccccc")


# table = table.reset_index(level=(0, 1))
# table = table.sort_values(by=[x_metric])
# generate_plot(table, metric, x_metric)
# table

# %%
# ************************************************************ #
# ********************* logp vs N ******************* #
# ************************************************************ #

criteria = [
    # "`name` == 'SO3'",
    # "`dataset/conditional` == False",
    # "`warmup_steps` == 100",
    # "`steps` == 100000",
    # "`dataset` == 32",
    # "`flow/beta_f` == 6",
    "`name` == 'ablation'",
    # "`optim/learning_rate` == 0.0002",
    "`flow/N` > 1",
    # "`loss` == 'dsm' & `loss/n_max` == 20 & `loss/thresh` == 0.5",
    "`loss` == 'ism'",
    # "`method` == 'RSGM'",
]
criteria = ["all"] if criteria == [] else criteria
results = query(runs_df, " & ".join(criteria))
results = results[results.group != "ablation_loss=ism,model=rsgm"]

x_metric = "flow/N"
metric = "test/logp"
val_metric = metric
group_by = [x_metric, "method"]

data, group_max_idx = select_hyperparms(results, group_by, metric, val_metric)

data = data.reset_index(level=(0, 1))
data = data.sort_values(by=[x_metric])

fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharey=True, sharex=True)
colors = sns.color_palette(cmap_name, 1)
fontsize = 30

x = data[x_metric]
y_low, y_mean, y_up = (
    data[metric]["lower_ci"],
    data[metric]["mean"],
    data[metric]["upper_ci"],
)
ax.plot(x, data[metric]["mean"], color=colors[0], lw=5)
ax.fill_between(x, y_low, y_up, alpha=0.2, facecolor=colors[0])
ax.tick_params(axis="both", which="major", labelsize=4 / 5 * fontsize)
ax.set_ylabel("Test log-likelihood", fontsize=fontsize)
ax.set_xlabel("Discretization steps N", fontsize=fontsize)
ax.set_xscale("log")
ax.set_xticks([2, 5, 10, 20, 50, 100, 200], [2, 5, 10, 20, 50, 100, 200])

# fig, axis = generate_plot(data, metric, x_metric)

fig_name = f"../doc/images/s2_ablation_N.pdf"
fig.savefig(fig_name, bbox_inches="tight", transparent=True)

data
# %%
# ************************************************************ #
# ********************* heatmap lr & beta_f ******************* #
# ************************************************************ #

criteria = [
    "`name` == 'SO3'",
    "`dataset/conditional` == False",
    "`steps` == 100000",
    "`warmup_steps` == 100",
    "`optim/learning_rate` >= 0.0001",
    "`optim/learning_rate` < 0.002",
    "`flow/beta_f` < 12",
    "`flow/beta_f` > 2",
    # "`flow/N` > 1",
    "`loss` == 'dsm'",
    # "`loss` == 'ism'",
    "`dataset` == 32",
    "(method != 'RSGM' | (method == 'RSGM' & `flow/N` == 100))",
    "(method == 'Moser Flow' | (method != 'Moser Flow' & `flow/beta_0` == 0.001))",
]
criteria = ["all"] if criteria == [] else criteria
results = query(runs_df, " & ".join(criteria))

results = results.rename(
    columns={
        "flow/beta_f": "$\\beta_f$",
        "optim/learning_rate": "Learning rate",
        "test/logp": "Test log-likelihood",
    }
)

x_metric = ["Learning rate", "$\\beta_f$"]
metric = "Test log-likelihood"
val_metric = "val/logp"
group_by = x_metric + ["method"]

data, _ = select_hyperparms(results, group_by, metric, val_metric)
#%%
# data = data.reset_index(level=(0))
data = data.sort_values(by=x_metric)
data = data.droplevel("group")
# data = data[(metric, "mean")]

# generate_plot(data, metric, x_metric)

x_metric_0_grid = sorted(list(data.index.get_level_values(x_metric[0]).unique()))
x_metric_1_grid = sorted(list(data.index.get_level_values(x_metric[1]).unique()))
method_grid = list(data.index.get_level_values("method").unique())
matrix = np.ones((len(method_grid), len(x_metric_0_grid), len(x_metric_1_grid))) * np.NAN

for k, (index_k, new_df) in enumerate(data.groupby(level=-1)):
    for i, (index_i, new_df) in enumerate(new_df.groupby(level=0)):
        for j, (beta_j, new_df) in enumerate(new_df.groupby(level=1)):
            value = new_df[(metric, "mean")]
            # print(index_k)
            matrix[
                method_grid.index(index_k),
                x_metric_0_grid.index(index_i),
                x_metric_1_grid.index(beta_j),
            ] = value
matrix = matrix.transpose((0, 2, 1))

fig, axis = plt.subplots(
    1,
    len(method_grid) + 1,
    gridspec_kw={"width_ratios": [1, 1, 0.08]},
    figsize=(10, 6),
)
axis[-1].figure.set_size_inches(10, 4.3)
fontsize = 25
# axis[0].get_shared_y_axes().join(axis[1])
# cmap = sns.color_palette("magma", as_cmap=True)
# cmap = sns.color_palette("viridis", as_cmap=True)
cmap = sns.cm.rocket
tick = ticker.ScalarFormatter(useOffset=False, useMathText=True)
tick.set_powerlimits((0, 0))
x_metric_0_grid = ["${}$".format(tick.format_data(x)) for x in x_metric_0_grid]

for i, method in enumerate(method_grid):
    g = sns.heatmap(
        matrix[i],
        xticklabels=x_metric_0_grid,
        yticklabels=x_metric_1_grid,
        label=method_grid[i],
        ax=axis[i],
        cbar=i == 1,
        cbar_ax=axis[-1] if i == 1 else None,
        square=True,
        cmap=cmap,
        linewidths=0.5,
        vmin=np.nanmin(matrix),
        vmax=np.nanmax(matrix),
        annot=True,
        fmt=".2f",
        annot_kws={"size": 3 / 4 * fontsize},
    )
    axis[i].set_xlabel(x_metric[0], fontsize=fontsize)
    if i == 0:
        axis[i].set_ylabel(
            x_metric[1], fontsize=1.2 * fontsize, rotation=0, labelpad=2 / 3 * fontsize
        )
    axis[i].tick_params(axis="both", which="major", labelsize=3 / 5 * fontsize)
    # axis[i].tick_params(axis="both", which="minor", labelsize=0.8 * 15)
    axis[i].set_title(f"{method}", fontsize=fontsize)
cbar = g.collections[0].colorbar
cbar.ax.tick_params(
    labelsize=2 / 3 * fontsize,
)
cbar.set_label(label=metric, fontsize=fontsize, labelpad=2 / 5 * fontsize)

fig.tight_layout(pad=-1.5)
fig_name = f"../doc/images/so3_heatmap.pdf"
fig.savefig(fig_name, bbox_inches="tight", transparent=True)

# data
# %%
# ************************************************************ #
# ********************* heatmap DSM loss   ******************* #
# ************************************************************ #

criteria = [
    "`name` == 'ablation'",
    "`loss` == 'dsm'",
    # "`loss` == 'ism'",
    "`flow/N` == 100",
]
criteria = ["all"] if criteria == [] else criteria
results = query(runs_df, " & ".join(criteria))

results = results.rename(
    columns={
        "loss/n_max": "$J$",
        "loss/thresh": "$\\tau$",
        "test/logp": "Test log-likelihood",
    }
)
# results.loc[:, "$J$"] = results.loc[:, "$J$"] + 1
results.loc[:, "$J$"] = results.loc[:, "$J$"].astype(int)
results.loc[:, "Negative log-likelihood"] = -results.loc[:, "Test log-likelihood"]

x_metric = ["$J$", "$\\tau$"]
metric = "Negative log-likelihood"
val_metric = "val/logp"
group_by = x_metric + ["method"]

data, _ = select_hyperparms(results, group_by, metric, val_metric, best=min)

data = data.sort_values(by=x_metric)
data = data.droplevel("group")

x_metric_0_grid = sorted(list(data.index.get_level_values(x_metric[0]).unique()))
x_metric_1_grid = sorted(
    list(data.index.get_level_values(x_metric[1]).unique()), reverse=True
)
method_grid = list(data.index.get_level_values("method").unique())
matrix = np.ones((len(method_grid), len(x_metric_0_grid), len(x_metric_1_grid))) * np.NAN

for k, (index_k, new_df) in enumerate(data.groupby(level=-1)):
    for i, (index_i, new_df) in enumerate(new_df.groupby(level=0)):
        for j, (index_j, new_df) in enumerate(new_df.groupby(level=1)):
            value = new_df[(metric, "mean")]
            matrix[
                method_grid.index(index_k),
                x_metric_0_grid.index(index_i),
                x_metric_1_grid.index(index_j),
            ] = value
matrix = matrix.transpose((0, 2, 1))
# thresh=1 is the same for all values of J
matrix[:, 0, :] = matrix[:, 0, 0]

fig, axis = plt.subplots(
    1,
    len(method_grid) + 1,
    gridspec_kw={"width_ratios": [1, 0.08 / 2]},
    figsize=(10, 6),
)
axis[-1].figure.set_size_inches(10, 4.3)
fontsize = 25
cmap = sns.cm.rocket_r
# cmap = sns.color_palette(cmap_name, as_cmap=True)

for i, method in enumerate(method_grid):
    g = sns.heatmap(
        matrix[i],
        xticklabels=x_metric_0_grid,
        yticklabels=x_metric_1_grid,
        label=method_grid[i],
        ax=axis[i],
        cbar=i == 0,
        cbar_ax=axis[-1] if i == 0 else None,
        square=True,
        cmap=cmap,
        linewidths=0.5,
        vmin=np.nanmin(matrix),
        vmax=np.nanmax(matrix),
        annot=True,
        fmt=".2f",
        annot_kws={"size": 3 / 4 * fontsize},
    )
    axis[i].set_xlabel(x_metric[0], fontsize=fontsize)
    if i == 0:
        axis[i].set_ylabel(
            x_metric[1], fontsize=1.2 * fontsize, rotation=0, labelpad=2 / 3 * fontsize
        )
    axis[i].tick_params(axis="both", which="major", labelsize=3 / 5 * fontsize)
    # axis[i].tick_params(axis="both", which="minor", labelsize=0.8 * 15)
    # axis[i].set_title(f"{method}", fontsize=fontsize)
cbar = g.collections[0].colorbar
cbar.ax.tick_params(
    labelsize=2 / 3 * fontsize,
)
cbar.set_label(label=metric, fontsize=fontsize, labelpad=2 / 5 * fontsize)

fig.tight_layout(pad=-3)
fig_name = f"../doc/images/s2_heatmap.pdf"
fig.savefig(fig_name, bbox_inches="tight", transparent=True)

# %%
