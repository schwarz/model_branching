import math
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("PDF")
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from pathlib import Path

heuristics = [
    "synced",
    "greedy_oracle",
    "best_last",
    "best_last_2",
    "best_last_3",
    "best_improving",
    "random",
    "local",
    "static",
    "global",  # joined in from global node
]

baselines = ["synced", "local", "static", "global"]

heuristic_names = {
    "synced": "Synced",
    "greedy_oracle": "Greedy Oracle",
    "best_last": "Best Last 1",
    "best_last_2": "Best Last 2",
    "best_last_3": "Best Last 3",
    "best_improving": "Best Improving",
    "random": "Random",
    "local": "Local",
    "static": "Static",
    "global": "Global",
}

# Lower value is always better for the used loss metrics
def times_better(series_a, series_b):
    """Returns a percentage value for how often the value of series a is lower than the respective value in series b"""
    assert len(series_a) == len(series_b), "series should have the same length"
    equal_count = better_count = worse_count = 0

    for i in range(len(series_a)):
        if series_a[i] > series_b[i]:
            worse_count = worse_count + 1
        if series_a[i] == series_b[i]:
            equal_count = equal_count + 1
        if series_a[i] < series_b[i]:
            better_count = better_count + 1
    return (
        better_count * 1.0 / len(series_a),
        equal_count * 1.0 / len(series_a),
        worse_count * 1.0 / len(series_a),
    )


experiments = [
    # "iid_lr0.1_le1_bs2",
    # "iid_lr0.1_le1_bs4",
    # "iid_lr0.1_le2_bs2",
    # "iid_lr0.1_le2_bs4",
    # "iid_lr0.5_le1_bs2",
    # "iid_lr0.5_le1_bs4",
    # "iid_lr0.5_le2_bs2",
    # "iid_lr0.5_le2_bs4",
    # non
    # "non_lr0.1_le1_bs2",
    # "non_lr0.1_le1_bs4",
    # "non_lr0.1_le2_bs2",
    # "non_lr0.1_le2_bs4",
    # "non_lr0.5_le1_bs2",
    # "non_lr0.5_le1_bs4",
    "non_lr0.5_le2_bs2",
    # "non_lr0.5_le2_bs4"
]

DATASET = "synthetic"  # demand fitrec_ds fitrec_hr synthetic turnstile
DATASET_SUFFIX = ""

if DATASET == "fitrec_ds":
    experiments = ["ds_" + e for e in experiments]
    DATASET = "fitrec"
    DATASET_SUFFIX = "_ds"
elif DATASET == "fitrec_hr":
    experiments = ["hr_" + e for e in experiments]
    DATASET = "fitrec"
    DATASET_SUFFIX = "_hr"

MEAN_PRECISIONS = {"demand": 3, "fitrec": 3, "synthetic": 5, "turnstile": 3}

SAVE_TABLES = False
SAVE_ZOOMED_FIGURES = True
SAVE_FIGURES = False
SAVE_SCATTERS = False
SAVE_BOXPLOTS = False

if __name__ == "__main__":
    iid_heuristic_improvements = {h: [] for h in heuristics}
    non_heuristic_improvements = {h: [] for h in heuristics}
    h_losses = {h: [] for h in heuristics}
    for current_experiment in experiments[:]:
        EXPERIMENT = current_experiment
        QUALIFIED_EXPERIMENT = DATASET + "/" + EXPERIMENT
        print("\n> " + QUALIFIED_EXPERIMENT)

        # Load log files
        acc_paths = list(
            Path("../experiments/{}".format(QUALIFIED_EXPERIMENT)).rglob(
                "acc_node_*.csv"
            )
        )
        acc_paths.sort()

        heu_paths = list(
            Path("../experiments/{}".format(QUALIFIED_EXPERIMENT)).rglob(
                "heu_node_*.csv"
            )
        )
        heu_paths.sort()

        df_pn = pd.read_csv(
            "../experiments/{}/acc_global.csv".format(QUALIFIED_EXPERIMENT)
        )
        df_pn["heuristic"] = df_pn[
            "model_key"
        ]  # sets the heuristic to "global" in all rows
        df_pn = df_pn[df_pn["iteration"] >= 3]  # no branching before this iteration

        print("Loading nodes")
        indexed_dfs = []
        for (acc_path, heu_path) in list(zip(acc_paths, heu_paths)):
            node = int(acc_path.name.split(".")[0].split("_")[-1])
            df_a = pd.read_csv(acc_path)
            df_a = df_a[df_a.iteration >= 3]
            df_h = pd.read_csv(heu_path)
            df_h = df_h[df_h.iteration >= 3]
            df_h["model_key"] = df_h["model"]
            del df_h["model"]

            df_global = df_pn[df_pn["id"] == node].drop("id", axis=1)

            df = pd.merge(df_a, df_h, on=["iteration", "model_key"]).drop(
                ["id_x", "id_y"], axis=1
            )

            whole_df = pd.concat([df, df_global])
            sorted_df = whole_df.sort_values(by=["iteration"])
            indexed_dfs.append((node, sorted_df))

        result_dfs = []  # Will become tables
        loss_dfs = []  # For further analysis purposes?
        improv_dfs = []  # Will be plotted

        heu_datas = []

        print("Processing nodes")
        for (node_id, df) in indexed_dfs:
            # Prune some heuristic traces
            longest = 0
            shortest = 1_000_000
            for h in heuristics:
                l = len(df[df.heuristic == h].loss.to_numpy())
                if l < shortest:
                    shortest = l
                if l > longest:
                    longest = l
            if shortest != longest:
                print(
                    node_id,
                    "Adjusted the lengths to match, shortest was",
                    shortest,
                    "and longest",
                    longest,
                )

            loss_df = pd.DataFrame(
                data={
                    h: df[df.heuristic == h].loss.to_numpy()[:shortest]
                    for h in heuristics
                }
            )
            loss_dfs.append(loss_df)

            for h in heuristics:
                h_losses[h].extend(loss_df[h].to_numpy())

            heu_data = {h: {} for h in heuristics}
            improv = {}
            for h in [h for h in heuristics]:
                heu_data[h]["heuristic"] = h
                heu_data[h]["median"] = np.median(loss_df[h].to_numpy())
                heu_data[h]["mean"] = np.mean(loss_df[h].to_numpy())
                heu_data[h]["std"] = np.std(loss_df[h].to_numpy())
                (
                    heu_data[h]["accuracy_better"],
                    heu_data[h]["accuracy_equal"],
                    heu_data[h]["accuracy_worse"],
                ) = times_better(loss_df[h], loss_df["synced"])

                # Branch length analysis
                dfg = df[(df["heuristic"] == h)].groupby(["model_key"])
                branches_len_1 = len([l for l in dfg.size() if l == 1])
                mean_length = dfg.size().mean()
                heu_data[h]["num_branches"] = len(dfg)
                heu_data[h]["portion_len_1"] = branches_len_1 / (len(dfg) * 1.0)
                heu_data[h]["mean_length"] = mean_length

                # Improvement
                loss_ratio = loss_df[h] / loss_df["synced"]
                improv[h] = ((1 - loss_ratio) * 100).to_numpy()
                # Depends on "synced" being processed first
                if h == "synced":
                    heu_data[h]["mean_improvement"] = 0.0
                else:
                    heu_data[h]["mean_improvement"] = (
                        1 - heu_data[h]["mean"] / heu_data["synced"]["mean"]
                    ) * 100
                if "non_" in EXPERIMENT:
                    non_heuristic_improvements[h].append(
                        heu_data[h]["mean_improvement"]
                    )
                else:
                    iid_heuristic_improvements[h].append(
                        heu_data[h]["mean_improvement"]
                    )

            result_dfs.append((node_id, pd.DataFrame(data=list(heu_data.values()))))
            improv_dfs.append((node_id, pd.DataFrame(data=improv)))
            heu_datas.append(heu_data)

        # Summary
        means = {}
        better = {}
        equal = {}
        worse = {}

        for d in heu_datas:
            for h, hd in d.items():
                if h in means:
                    means[h].append(hd["mean"])
                    better[h].append(hd["accuracy_better"])
                    equal[h].append(hd["accuracy_equal"])
                    worse[h].append(hd["accuracy_worse"])
                else:
                    means[h] = [hd["mean"]]
                    better[h] = [hd["accuracy_better"]]
                    equal[h] = [hd["accuracy_equal"]]
                    worse[h] = [hd["accuracy_worse"]]

        summary_df = pd.DataFrame(
            columns=[
                "heuristic",
                "mean",
                "mean_improvement",
                "accuracy_better",
                "accuracy_equal",
                "accuracy_worse",
            ]
        )
        for h in heuristics:
            summary_df = summary_df.append(
                {
                    "heuristic": h,
                    "mean": np.mean(means[h]),
                    "mean_improvement": (
                        1 - np.mean(means[h]) / np.mean(means["synced"])
                    )
                    * 100,
                    "accuracy_better": np.mean(better[h]),
                    "accuracy_equal": np.mean(equal[h]),
                    "accuracy_worse": np.mean(worse[h]),
                },
                ignore_index=True,
            )
        if SAVE_ZOOMED_FIGURES:
            if not os.path.isdir("graphics/results_zoomed/" + DATASET):
                os.mkdir("graphics/results_zoomed/" + DATASET)
            if not os.path.isdir(
                "graphics/results_zoomed/" + DATASET + "/" + EXPERIMENT
            ):
                os.mkdir("graphics/results_zoomed/" + DATASET + "/" + EXPERIMENT)

            print("Generating zoomed iteration improvement plots")
            plt.ioff()
            for (node_id, df) in improv_dfs:
                print("Node", node_id)
                for h in df.columns:
                    if h in baselines and False:  # Undecided
                        continue
                    h_name = h.replace("_", " ").title()

                    plot_df = df
                    for i in range(2):
                        plot_df = plot_df.rolling(2).mean()
                        plot_df = plot_df.iloc[::2, :]

                    plot_df = plot_df[h].reset_index()

                    # https://ercanozturk.org/2017/12/16/python-matplotlib-plots-in-latex/
                    f = plt.figure()
                    plt.bar(
                        np.arange(len(plot_df)),
                        plot_df[h],
                        color=(plot_df[h] >= 0).map(
                            {True: "#228be6", False: "#e64980"}
                        ),
                    )

                    plt.rc("text", usetex=True)
                    plt.rc("font", family="serif")
                    # set labels (LaTeX can be used)
                    plt.title("\\textbf{" + h_name + "}", fontsize=11)
                    plt.xlabel(r"\textbf{Iteration Group}", fontsize=11)
                    plt.ylabel(r"\textbf{\% Improvement}", fontsize=11)

                    # save as PDF
                    f.savefig(
                        "graphics/results_zoomed/{}/{}/n{}_{}.pdf".format(
                            DATASET, EXPERIMENT, node_id, h
                        ),
                        bbox_inches="tight",
                    )
                    plt.close(f)

        if SAVE_FIGURES:
            if not os.path.isdir("graphics/results/" + DATASET):
                os.mkdir("graphics/results/" + DATASET)
            if not os.path.isdir("graphics/results/" + DATASET + "/" + EXPERIMENT):
                os.mkdir("graphics/results/" + DATASET + "/" + EXPERIMENT)

            print("Generating iteration improvement plots")
            plt.ioff()
            for (node_id, df) in improv_dfs:
                print("Node", node_id)
                for h in df.columns:
                    if h in baselines and False:  # Undecided
                        continue
                    h_name = h.replace("_", " ").title()
                    # https://ercanozturk.org/2017/12/16/python-matplotlib-plots-in-latex/
                    f = plt.figure()
                    plt.bar(
                        range(0, len(df)),
                        df[h],
                        color=(df[h] >= 0).map({True: "#228be6", False: "#e64980"}),
                    )

                    plt.rc("text", usetex=True)
                    plt.rc("font", family="serif")
                    # set labels (LaTeX can be used)
                    plt.title("\\textbf{" + h_name + "}", fontsize=11)
                    plt.xlabel(r"\textbf{Iteration}", fontsize=11)
                    plt.ylabel(r"\textbf{\% Improvement}", fontsize=11)

                    # save as PDF
                    f.savefig(
                        "graphics/results/{}/{}/n{}_{}.pdf".format(
                            DATASET, EXPERIMENT, node_id, h
                        ),
                        bbox_inches="tight",
                    )
                    plt.close(f)

        tex_dfs = result_dfs + [("Summary", summary_df)]
        for (_, df) in tex_dfs:
            # Combine accuracy_{better, equal, worse} into one string column
            df["comparison"] = df[
                ["accuracy_better", "accuracy_equal", "accuracy_worse"]
            ].apply(
                lambda x: "{:.0f} / {:.0f} / {:.0f}".format(
                    100 * x[0], 100 * x[1], 100 * x[2]
                ),
                axis=1,
            )
            # Rename heuristics
            df["heuristic"] = df["heuristic"].str.replace("_", " ").str.title()
            # Float rounding # TODO Can be removed?
            df["mean_improvement"] = df["mean_improvement"].round(decimals=2)

        if not os.path.isdir("tables/" + DATASET):
            os.mkdir("tables/" + DATASET)
        if not os.path.isdir("tables/" + DATASET + "/" + EXPERIMENT):
            os.mkdir("tables/" + DATASET + "/" + EXPERIMENT)

        if SAVE_TABLES:
            for (node_id, df) in tex_dfs:
                prec = MEAN_PRECISIONS[DATASET]
                table_df = df.round({"mean": prec})
                tbl = table_df.to_latex(
                    columns=["heuristic", "mean", "comparison", "mean_improvement"],
                    header=[
                        "\\textbf{Heuristic}",
                        "\\textbf{MAE ↓}",
                        "\\textbf{\\% Be/Eq/Wo}",
                        "\\textbf{\\% Improv.}",
                    ],
                    caption="Node " + str(node_id),
                    label="tab:{}_{}".format(EXPERIMENT.replace(".", ""), node_id),
                    bold_rows=False,
                    index=False,
                    escape=False,
                )
                lines = tbl.split("\n")
                table = lines[4:-2] + lines[2:4]
                # Remove captions of summary node tables, set manually later
                if node_id == "Summary":
                    table[-2] = ""  # No caption for the summary node
                pre_lines = table[0:4]
                content_lines = table[4 : 4 + 10]
                post_lines = table[4 + 10 :]

                if node_id == "Summary":
                    content_lines.append(content_lines.pop(8))

                    # Move greedy oracle past regular heuristics
                    content_lines.insert(6, content_lines[1])
                    content_lines.pop(1)

                    # Remove Best Last 2
                    content_lines.pop(2)
                    # Remove Best Last 1
                    content_lines.pop(1)

                    # nsmallest(2)
                    smallest_2_maes = table_df[
                        (table_df["heuristic"] != "Best Last")
                        & (table_df["heuristic"] != "Best Last 2")
                    ]["mean"].nsmallest(2)
                    for i in range(len(content_lines)):
                        smallest_1 = "{:.{}f}".format(
                            smallest_2_maes.to_numpy()[0], prec
                        )
                        smallest_2 = "{:.{}f}".format(
                            smallest_2_maes.to_numpy()[1], prec
                        )
                        if smallest_1 in content_lines[i]:
                            content_lines[i] = content_lines[i].replace(
                                smallest_1, "\\textbf{" + smallest_1 + "}"
                            )

                        if smallest_2 in content_lines[i]:
                            content_lines[i] = content_lines[i].replace(
                                smallest_2, "\\underline{" + smallest_2 + "}"
                            )

                    # Add midrules after synced, random and before global
                    content_lines.insert(7, "\\midrule")
                    content_lines.insert(5, "\\midrule")
                    content_lines.insert(1, "\\midrule")

                    content_lines.insert(4, "\\addlinespace")

                with open(
                    "tables/{}/{}/node{}.tex".format(DATASET, EXPERIMENT, node_id), "w"
                ) as f:
                    f.write("\n".join(pre_lines + content_lines + post_lines) + "\n")

    if SAVE_SCATTERS:
        print("Generating scatter plots for all")
        if not os.path.isdir(
            "graphics/results/" + DATASET + DATASET_SUFFIX + "/scatters"
        ):
            os.mkdir("graphics/results/" + DATASET + DATASET_SUFFIX + "/scatters")

        for hc in ["static", "local", "global"]:
            for h in heuristics:
                # skip baselines
                if h in baselines:
                    continue
                f = plt.figure()
                plt.plot(
                    iid_heuristic_improvements[hc],
                    iid_heuristic_improvements[h],
                    ".",
                    label="IID",
                    color="#228be6",
                )

                plt.plot(
                    non_heuristic_improvements[hc],
                    non_heuristic_improvements[h],
                    ".",
                    label="Non-IID",
                    color="#e64980",
                )

                plt.rc("text", usetex=True)
                plt.rc("font", family="serif")
                # set labels (LaTeX can be used)
                # plt.title("\\textbf{" + h_name + "}", fontsize=11)
                plt.xlabel("\\textbf{" + heuristic_names[hc] + "}", fontsize=11)
                plt.ylabel("\\textbf{" + heuristic_names[h] + "}", fontsize=11)
                plt.legend()
                # save as PDF
                f.savefig(
                    "graphics/results/{}/scatters/{}_{}.pdf".format(
                        DATASET + DATASET_SUFFIX,
                        heuristic_names[hc],
                        heuristic_names[h],
                    ),
                    bbox_inches="tight",
                )
                plt.close(f)
    # Aggregate all nodes from all experiments
    if SAVE_BOXPLOTS:
        print(list(h_losses.keys()))
        bp_df = pd.DataFrame(data=h_losses)
        bp_df = bp_df.rename(
            columns={
                "synced": "Synced",
                "best_last_3": "Best Last 3",
                "best_improving": "Best Improving",
                "local": "Local",
            }
        )
        print(bp_df.head())

        print("Generating boxplots for all")
        if not os.path.isdir(
            "graphics/results/" + DATASET + DATASET_SUFFIX + "/boxplots"
        ):
            os.mkdir("graphics/results/" + DATASET + DATASET_SUFFIX + "/boxplots")
        f = plt.figure()
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        # set labels (LaTeX can be used)
        title = DATASET
        if DATASET == "fitrec":
            title = "FitRec " + DATASET_SUFFIX[1:].upper()
        else:
            title = title.capitalize()
        plt.title("\\textbf{" + title + "}", fontsize=11)
        plt.xlabel("\\textbf{" + "Strategy" + "}", fontsize=11)
        plt.ylabel("\\textbf{" + "Loss" + "}", fontsize=11)
        bp = bp_df.boxplot(column=["Synced", "Best Last 3", "Best Improving", "Local"])
        # save as PDF
        f.savefig(
            "graphics/results/{}/boxplots/{}.pdf".format(
                DATASET + DATASET_SUFFIX, "boxplot"
            ),
            bbox_inches="tight",
        )
        plt.close(f)
