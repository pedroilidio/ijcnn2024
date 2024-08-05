import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp


def _find_maximal_cliques(adj_matrix):
    """Wrapper function over the recursive Bron-Kerbosch algorithm.
    Will be used to find points that are under the same crossbar.
    Parameters
    ----------
    adj_matrix : pd.DataFrame
        Matrix containing 1 if row item and column item do NOT significantly
        differ. Diagonal must be zeroed.
    Returns
    -------
    list[set]
        Largest fully conected subgraphs.
    """
    return _bron_kerbosch(set(), set(adj_matrix.index), set(), adj_matrix)


def _bron_kerbosch(R, P, X, adj_matrix):
    """Recursive algrithm to find the maximal fully connected subgraphs.
    See https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
    adj_matrix : pd.DataFrame
        Matrix containing 1 if row item and column item do NOT significantly
        differ. Diagonal must be zeroed.
    Returns
    -------
    list[set]
        Largest fully conected subgraphs.
    """
    if len(P) == 0 and len(X) == 0:
        return [R]
    res = []
    for v in P.copy():
        N = {n for n in adj_matrix.index if adj_matrix.loc[v, n]}
        res += _bron_kerbosch(R | {v}, P & N, X & N, adj_matrix)
        P.remove(v)
        X.add(v)
    return res


def plot_critical_difference_diagram(
    ranks: dict | pd.Series,
    sig_matrix: pd.DataFrame,
    *,
    ax = None,
    label_fmt_left: str = '{label} ({rank:.2g})',
    label_fmt_right: str = '({rank:.2g}) {label}',
    label_props: dict = None,
    marker_props: dict = None,
    elbow_props: dict = None,
    crossbar_props: dict = None,
    text_h_margin: float = 0.01
) -> dict[str, list]:
    """Plot a Critical Difference diagram from ranks and post-hoc results.

    The diagram arranges the average ranks of multiple groups on the x axis
    in order to facilitate performance comparisons between them. The groups
    that could not be statistically deemed as different are linked by a
    horizontal crossbar.

                       rank markers
         X axis ---------O----O-------------------O-O------------O---------
                         |----|                   | |            |   
                         |    |                   |---crossbar---|   
                clf1 ----|    |                   | |            |---- clf3
                clf2 ---------|                   | |----------------- clf4
                                                  |------------------- clf5
                    |____|
                text_h_margin

    In the drawing above, the two crossbar indicates that clf1 and clf2 cannot
    be statistically differentiated, the same occurring between clf3, clf4 and
    clf5. However, clf1 and clf2 are each significantly lower ranked than clf3,
    clf4 and clf5.

    Parameters
    ----------
    ranks : dict or series
        Indicates the rank value for each sample or estimator (as keys or index).
    sig_matrix : pd.DataFrame
        The corresponding p-value matrix output by post-hoc tests, with
        indices matching the labels in the ranks argument.
    ax : matplotlib.Axes, optional
        The Axes object in which the plot will be built. Gets the current Axes
        by default (if None is passed).
    label_fmt_left : str, optional
        The format string to apply to the labels on the left side. The keywords
        label and rank can be used to specify the sample/estimator name and
        rank value, respectively, by default '{label} ({rank:.2g})'.
    label_fmt_right : str, optional
        The same, but for the labels on the right side of the plot.
        By default '({rank:.2g}) {label}'.
    label_props : dict, optional
        Parameters to be passed to pyplot.annotate() when creating the labels,
        by default None.
    marker_props : dict, optional
        Parameters to be passed to pyplot.scatter() when plotting the rank
        markers on the axis, by default None.
    elbow_props : dict, optional
        Parameters to be passed to pyplot.plot() when creating the elbow lines,
        by default None.
    crossbar_props : dict, optional
        Parameters to be passed to pyplot.plot() when creating the crossbars
        that indicate lack of statistically significant difference. By default
        None.
    text_h_margin : float, optional
        Space between the text labels and the nearest vertical line of an
        elbow, by default 0.01.

    Returns
    -------
    dict[str, list[matplotlib.Artist]]
        Lists of Artists created.

    References
    ----------
    [1] Demšar, J. (2006). Statistical comparisons of classifiers over multiple
    data sets. The Journal of Machine learning research, 7, 1-30.

    [2] https://mirkobunse.github.io/CriticalDifferenceDiagrams.jl/stable/
    """
    elbow_props = elbow_props or {}
    marker_props = dict(zorder=3) | (marker_props or {})
    label_props = dict(va='center') | (label_props or {})
    crossbar_props = dict(color='k', zorder=3, linewidth=2) | (
        crossbar_props or {})

    ax = ax or plt.gca()
    ax.yaxis.set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('top')
    ax.spines['top'].set_position('zero')

    # lists of artitsts to be output
    markers = []
    elbows = []
    labels = []
    crossbars = []

    # True if pairwise comparison is NOT significant
    adj_matrix = pd.DataFrame(
        1 - sp.sign_array(sig_matrix),
        index=sig_matrix.index,
        columns=sig_matrix.columns,
        dtype=bool,
    )

    ranks = pd.Series(ranks)  # Standardize if ranks is dict
    points_left, points_right = np.array_split(ranks.sort_values(), 2)

    # Create stacking of crossbars: for each level, try to fit the crossbar,
    # so that it does not intersect with any other in the level. If it does not
    # fit in any level, create a new level for it.
    crossbar_levels: list[list[set]] = []
    for bar in _find_maximal_cliques(adj_matrix):
        for level, bars_in_level in enumerate(crossbar_levels):
            if not any(bool(bar & bar_in_lvl) for bar_in_lvl in bars_in_level):
                ypos = -level-1
                bars_in_level.append(bar)
                break
        else:
            ypos = -len(crossbar_levels) - 1
            crossbar_levels.append([bar])

        crossbars.append(ax.plot(
            [ranks[i] for i in bar],
            [ypos] * len(bar),
            **crossbar_props,
        ))

    lowest_crossbar_ypos = -len(crossbar_levels)

    def plot_items(points, xpos, label_fmt, label_props):
        """Plot each marker + elbow + label."""
        ypos = lowest_crossbar_ypos - 1
        for label, rank in points.items():
            elbow, *_ = ax.plot(
                [xpos, rank, rank],
                [ypos, ypos, 0],
                **elbow_props,
            )
            elbows.append(elbow)
            curr_color = elbow.get_color()
            markers.append(
                ax.scatter(rank, 0, **dict(color=curr_color) | marker_props)
            )
            labels.append(
                ax.annotate(
                    label_fmt.format(label=label, rank=rank),
                    (xpos, ypos),
                    **dict(color=curr_color) | label_props,
                )
            )
            ypos -= 1

    plot_items(
        points_left,
        xpos=points_left.iloc[0] - text_h_margin,
        label_fmt=label_fmt_left,
        label_props=dict(ha='right') | label_props,
    )
    plot_items(
        points_right[::-1],
        xpos=points_right.iloc[-1] + text_h_margin,
        label_fmt=label_fmt_right,
        label_props=dict(ha='left') | label_props,
    )

    return dict(
        markers=markers,
        elbows=elbows,
        labels=labels,
        crossbars=crossbars,
    )
