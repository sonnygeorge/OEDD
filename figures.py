from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


P1 = "1P"
P2 = "2P"
P1_RH = "1P+RH"
P2_RH = "2P+RH"
CHARS_PER_TOKEN = 4


def get_scores_for_length_class(
    df: pd.DataFrame, length_class: Literal["short", "medium", "long"]
) -> dict:
    groupby = df[df["length_class"] == length_class].groupby(
        [
            "chat_model_string",
            "include_red_herring",
            "require_intermediate_inference",
            "test_title",
        ]
    )
    groupby = (
        groupby.agg({"was_correct": "mean"})
        .reset_index()
        .groupby(
            [
                "chat_model_string",
                "include_red_herring",
                "require_intermediate_inference",
            ]
        )
    )
    df = groupby.agg({"was_correct": "mean"}).reset_index()
    data = {}
    for chat_model_string, group in df.groupby("chat_model_string"):
        model_data = {
            P2_RH: group[
                (group["include_red_herring"] == True)
                & (group["require_intermediate_inference"])
                == True
            ]["was_correct"].iloc[0],
            P2: group[
                (group["include_red_herring"] == False)
                & (group["require_intermediate_inference"] == True)
            ]["was_correct"].iloc[0],
            P1_RH: group[
                (group["include_red_herring"] == True)
                & (group["require_intermediate_inference"] == False)
            ]["was_correct"].iloc[0],
            P1: group[
                (group["include_red_herring"] == False)
                & (group["require_intermediate_inference"] == False)
            ]["was_correct"].iloc[0],
        }
        data[chat_model_string] = model_data
    return data


def get_length_token_bounds(df: pd.DataFrame, length_class: str) -> Tuple[int, int]:
    lower = df[df["length_class"] == length_class]["total_length"].min()
    upper = df[df["length_class"] == length_class]["total_length"].max()
    return int(lower / CHARS_PER_TOKEN), int(upper / CHARS_PER_TOKEN)


def get_barcharts(df: pd.DataFrame, show_a_priori_bias: bool = False):
    plt.clf()
    plt.style.use("default")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["font.family"] = "sans-serif"
    title_short = "Short Length ({lower}-{upper} Tokens)"
    title_medium = "Medium Length ({lower}-{upper} Tokens)"
    title_long = "Long Length ({lower}-{upper} Tokens)"

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(13, 8))

    for ax, length_class, title in zip(
        axes, ["short", "medium", "long"], [title_short, title_medium, title_long]
    ):
        lower, upper = get_length_token_bounds(df, length_class)
        title = title.format(lower=lower, upper=upper)
        data = get_scores_for_length_class(df, length_class)
        groups = list(data.keys())
        chat_model_string = [P1, P2, P1_RH, P2_RH]
        bar_values = [data[group] for group in groups]

        ax.axhline(0.5, linestyle=":", color="red", label="Random Baseline", zorder=1)
        bar_width = 0.2
        index = np.arange(len(groups))
        for i, label in enumerate(chat_model_string):
            values = [group[label] for group in bar_values]
            bars = ax.bar(index + i * bar_width, values, bar_width, label=label)
            # Annotate bars with values
            for bar in bars:
                height = bar.get_height()
                text = ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                )
                # Add white glow outline to text
                text.set_path_effects(
                    [path_effects.withStroke(linewidth=4, foreground="white")]
                )
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(Better Choice)")
        ax.set_title(title, pad=10, fontsize=11)
        ax.set_xticks(index + bar_width * 1.5)
        ax.set_xticklabels(groups)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, fancybox=True)

    plt.tight_layout()
    plt.savefig("barcharts.pdf", format="pdf")


def get_heatmaps(df: pd.DataFrame):
    plt.clf()
    plt.style.use("default")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["font.family"] = "sans-serif"
    # title = "P(Better Choice) Matrices by Model"
    models = df["chat_model_string"].unique()
    n_heatmaps = len(models)

    # Create GridSpec layout
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, n_heatmaps + 1, width_ratios=[1] * n_heatmaps + [0.1])

    cmap = LinearSegmentedColormap.from_list(
        "red_green", ["red", "yellow", "green"], N=256
    )
    if n_heatmaps == 1:
        axes = [axes]

    row_labels = [P1, P2, P1_RH, P2_RH]
    col_labels = ["Short", "Medium", "Long"]

    for i, chat_model_string in enumerate(models):
        # Create heatmap axes
        ax = fig.add_subplot(gs[0, i])
        model_df = df[df["chat_model_string"] == chat_model_string]
        short = get_scores_for_length_class(model_df, "short")[chat_model_string]
        medium = get_scores_for_length_class(model_df, "medium")[chat_model_string]
        long = get_scores_for_length_class(model_df, "long")[chat_model_string]
        data = [
            [short[P1], medium[P1], long[P1]],
            [short[P2], medium[P2], long[P2]],
            [short[P1_RH], medium[P1_RH], long[P1_RH]],
            [short[P2_RH], medium[P2_RH], long[P2_RH]],
        ]
        data = np.array(data)
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)

        ax.set_xlabel(chat_model_string, labelpad=10, fontsize=12)
        ax.xaxis.set_label_position("bottom")

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.xaxis.set_ticks_position("top")

        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.tick_params(top=False, bottom=False, left=False, right=False)

        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(
                    j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black"
                )
                text.set_path_effects(
                    [path_effects.withStroke(linewidth=2.5, foreground="white")]
                )

    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("P(Better Choice)")
    cbar.set_ticks([0.0, 0.5, 1.0])
    fig.tight_layout()
    fig.savefig("heatmaps.pdf", format="pdf")


def get_lineplot(df: pd.DataFrame, red_herring: Optional[bool] = None):
    plt.clf()
    plt.style.use("ggplot")
    df = df.copy()
    if red_herring is not None:
        df = df[df["include_red_herring"] == red_herring]
    df["premise_separation_tokens"] = (
        (
            (
                df["intermediate_inference_premise_one_prompt_span_end"]
                - df["intermediate_inference_premise_two_prompt_span_start"]
            )
            / CHARS_PER_TOKEN
        )
        .abs()
        .round(0)
    )
    bins = [50, 250, 400, 550, 700, 900, 1200, 1800, 3000]
    x_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    df["premise_separation_tokens_bins"] = pd.cut(
        df["premise_separation_tokens"],
        bins=bins,
        include_lowest=True,
    )

    plt.axhline(0.5, linestyle=":", color="red", label="Random Baseline")
    plt.ylim(0, 1)
    for chat_model_string, group in df.groupby("chat_model_string"):
        scores_by_bin = (
            group.groupby("premise_separation_tokens_bins", observed=True)
            .agg({"was_correct": "mean"})
            .reset_index()
        )
        plt.plot(
            scores_by_bin["was_correct"],
            label=chat_model_string,
            marker="o",
        )
    plt.xticks(range(len(x_labels)), x_labels, rotation=45)
    plt.xlabel("Premise Separation Tokens")
    plt.ylabel("P(Better Choice)")
    plt.legend(loc="lower left", bbox_to_anchor=(1, 0.5))
    if red_herring:
        plt.title("With Red Herring")
        fname = "lineplot_with_red_herring.pdf"
    elif red_herring is None:
        fname = "lineplot.pdf"
    else:
        plt.title("Without Red Herring")
        fname = "lineplot_without_red_herring.pdf"
    plt.tight_layout()
    plt.savefig(fname, format="pdf")


def get_pval_of_results_given_null_that_easier_setup_is_actually_harder(
    df_easier: pd.DataFrame, df_harder: pd.DataFrame
) -> float:
    """Given two DataFrames containing trial results for purportedly easier (higher
    success rate) and harder (lower success rate) bernoulli experiments, computes the
    p-value for a null hypothesis of the reverse being true--that is, the
    "easier" bernoulli experiment actually being the harder one.

    Args:
        df_easier (pd.DataFrame): A DataFrame containing trial results for the
            'easier' setup. Should have columns 'test_title' and 'was_correct',
            where 'was_correct' is a boolean indicating if the trial was successful.
        df_harder (pd.DataFrame): A DataFrame containing trial results for the
            'harder' setup. Same structure as `df_easier`.

    Returns:
        float: The p-value of the null hypothesis that the purportedly easier setup is,
        in fact, harder than the purportedly harder setup.
    """
    n_easier_tests = len(df_easier["test_title"].unique())
    n_harder_tests = len(df_harder["test_title"].unique())
    assert n_easier_tests == n_harder_tests

    n_easier_trials = df_easier.groupby("test_title").size().min() * n_easier_tests
    easier_p = (
        df_easier.groupby("test_title")
        .agg({"was_correct": "mean"})
        .mean()["was_correct"]
    )

    n_harder_trials = df_harder.groupby("test_title").size().min() * n_harder_tests
    harder_p = (
        df_harder.groupby("test_title")
        .agg({"was_correct": "mean"})
        .mean()["was_correct"]
    )

    _, p_val = sm.stats.proportions_ztest(
        count=[
            np.floor(n_easier_trials * easier_p),
            np.ceil(n_harder_trials * harder_p),
        ],
        nobs=[n_easier_trials, n_harder_trials],
        alternative="larger",
    )
    return p_val


def get_pval_of_null_that_each_test_is_greater_than_half(df: pd.DataFrame) -> float:
    """Calculates the p-value for the null hypothesis that the actual success rate is
    greater than or equal to 0.5.

    Args:
        df (pd.DataFrame): A DataFrame containing trial results. Should have columns
        'test_title' and 'was_correct',

    Returns:
        float: The p-value across all tests for the null hypothesis that the true success
        rate is greater than or equal to 0.5.
    """
    p_vals = []

    for test_title, group in df.groupby("test_title"):
        successes = group["was_correct"].sum()
        trials = len(group)
        p_val = stats.binom_test(successes, trials, p=0.5, alternative="greater")
        p_vals.append(p_val)

    return sum(p_vals) / len(p_vals)  # Law of total probability


def compute_paired_t_test_p_values(df: pd.DataFrame, chat_model_string: str) -> None:
    df = df[df["chat_model_string"] == chat_model_string]
    dfs_by_setup = {
        f"{P1} short": df[
            (df["include_red_herring"] == False)
            & (df["require_intermediate_inference"] == False)
            & (df["length_class"] == "short")
        ],
        f"{P1} medium": df[
            (df["include_red_herring"] == False)
            & (df["require_intermediate_inference"] == False)
            & (df["length_class"] == "medium")
        ],
        f"{P1} long": df[
            (df["include_red_herring"] == False)
            & (df["require_intermediate_inference"] == False)
            & (df["length_class"] == "long")
        ],
        f"{P2} short": df[
            (df["include_red_herring"] == False)
            & (df["require_intermediate_inference"] == True)
            & (df["length_class"] == "short")
        ],
        f"{P2} medium": df[
            (df["include_red_herring"] == False)
            & (df["require_intermediate_inference"] == True)
            & (df["length_class"] == "medium")
        ],
        f"{P2} long": df[
            (df["include_red_herring"] == False)
            & (df["require_intermediate_inference"] == True)
            & (df["length_class"] == "long")
        ],
        f"{P1_RH} short": df[
            (df["include_red_herring"] == True)
            & (df["require_intermediate_inference"] == False)
            & (df["length_class"] == "short")
        ],
        f"{P1_RH} medium": df[
            (df["include_red_herring"] == True)
            & (df["require_intermediate_inference"] == False)
            & (df["length_class"] == "medium")
        ],
        f"{P1_RH} long": df[
            (df["include_red_herring"] == True)
            & (df["require_intermediate_inference"] == False)
            & (df["length_class"] == "long")
        ],
        f"{P2_RH} short": df[
            (df["include_red_herring"] == True)
            & (df["require_intermediate_inference"] == True)
            & (df["length_class"] == "short")
        ],
        f"{P2_RH} medium": df[
            (df["include_red_herring"] == True)
            & (df["require_intermediate_inference"] == True)
            & (df["length_class"] == "medium")
        ],
        f"{P2_RH} long": df[
            (df["include_red_herring"] == True)
            & (df["require_intermediate_inference"] == True)
            & (df["length_class"] == "long")
        ],
    }

    red_herring_comparisons = [
        (f"{P1} short", f"{P1_RH} short"),
        (f"{P1} medium", f"{P1_RH} medium"),
        (f"{P1} long", f"{P1_RH} long"),
        (f"{P2} short", f"{P2_RH} short"),
        (f"{P2} medium", f"{P2_RH} medium"),
        (f"{P2} long", f"{P2_RH} long"),
    ]
    disparate_premises_comparisons = [
        (f"{P1} short", f"{P2} short"),
        (f"{P1} medium", f"{P2} medium"),
        (f"{P1} long", f"{P2} long"),
        (f"{P1_RH} short", f"{P2_RH} short"),
        (f"{P1_RH} medium", f"{P2_RH} medium"),
        (f"{P1_RH} long", f"{P2_RH} long"),
    ]
    short_to_medium_comparisons = [
        (f"{P1} short", f"{P1} medium"),
        (f"{P1_RH} short", f"{P1_RH} medium"),
        (f"{P2} short", f"{P2} medium"),
        (f"{P2_RH} short", f"{P2_RH} medium"),
    ]
    medium_to_long_comparisons = [
        (f"{P1} medium", f"{P1} long"),
        (f"{P1_RH} medium", f"{P1_RH} long"),
        (f"{P2} medium", f"{P2} long"),
        (f"{P2_RH} medium", f"{P2_RH} long"),
    ]

    print(f"\nModel: {chat_model_string}")
    print("\nAdding Red Herring:")
    for red_herring_comparison in red_herring_comparisons:
        df_easier = dfs_by_setup[red_herring_comparison[0]]
        df_harder = dfs_by_setup[red_herring_comparison[1]]
        p_val = get_pval_of_results_given_null_that_easier_setup_is_actually_harder(
            df_easier, df_harder
        )
        p_val_string = f"{p_val:.2E}" if p_val < 0.05 else f"{p_val:.2f}"
        print(
            f"  P('{red_herring_comparison[0]}' harder than '{red_herring_comparison[1]}'): {p_val_string}"
        )

    print("\nRequiring Reasoning Over Disparate Premises:")
    for disparate_premises_comparison in disparate_premises_comparisons:
        df_easier = dfs_by_setup[disparate_premises_comparison[0]]
        df_harder = dfs_by_setup[disparate_premises_comparison[1]]
        p_val = get_pval_of_results_given_null_that_easier_setup_is_actually_harder(
            df_easier, df_harder
        )
        p_val_string = f"{p_val:.2E}" if p_val < 0.05 else f"{p_val:.2f}"
        print(
            f"  P('{disparate_premises_comparison[0]}' harder than '{disparate_premises_comparison[1]}'): {p_val_string}"
        )

    print("\nGoing From Short to Medium:")
    for short_to_medium_comparison in short_to_medium_comparisons:
        df_easier = dfs_by_setup[short_to_medium_comparison[0]]
        df_harder = dfs_by_setup[short_to_medium_comparison[1]]
        p_val = get_pval_of_results_given_null_that_easier_setup_is_actually_harder(
            df_easier, df_harder
        )
        p_val_string = f"{p_val:.2E}" if p_val < 0.05 else f"{p_val:.2f}"
        print(
            f"  P('{short_to_medium_comparison[0]}' harder than '{short_to_medium_comparison[1]}'): {p_val_string}"
        )

    print("\nGoing From Medium to Long:")
    for medium_to_long_comparison in medium_to_long_comparisons:
        df_easier = dfs_by_setup[medium_to_long_comparison[0]]
        df_harder = dfs_by_setup[medium_to_long_comparison[1]]
        p_val = get_pval_of_results_given_null_that_easier_setup_is_actually_harder(
            df_easier, df_harder
        )
        p_val_string = f"{p_val:.2E}" if p_val < 0.05 else f"{p_val:.2f}"
        print(
            f"  P('{medium_to_long_comparison[0]}' harder than '{medium_to_long_comparison[1]}'): {p_val_string}"
        )


if __name__ == "__main__":
    df = pd.read_csv("results.csv")

    get_heatmaps(df)
    get_barcharts(df)
    get_lineplot(df, red_herring=True)
    get_lineplot(df, red_herring=False)
    get_lineplot(df)

    compute_paired_t_test_p_values(df, chat_model_string="gpt-3.5-turbo-0125")
    compute_paired_t_test_p_values(df, chat_model_string="gpt-4o-2024-05-13")
    compute_paired_t_test_p_values(df, chat_model_string="models/gemini-1.5-pro-latest")
