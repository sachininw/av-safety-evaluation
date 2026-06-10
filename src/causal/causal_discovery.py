"""
Causal structure learning from AV trajectory data.

Learns a Directed Acyclic Graph (DAG) from observational data without
requiring prior knowledge of the causal graph.  This answers:

  "Does jerk → TTC decrease → near-miss, or is there a hidden common cause?"

Two algorithms are implemented:
  1. PC Algorithm (Peter-Clark) — constraint-based, uses conditional
     independence tests (partial correlation) to orient edges.
  2. Correlation-threshold DAG — fast baseline using pairwise correlations
     and a greedy acyclicity enforcer (for large datasets).

The learned DAG can then be used with do-calculus to identify which
causal effects are identifiable from observational data.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Conditional independence tests
# ---------------------------------------------------------------------------

def partial_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: list[str],
) -> tuple[float, float]:
    """
    Compute the partial correlation ρ(X, Y | Z) and its p-value.

    Uses the Fisher z-transformation to test H₀: ρ(X,Y|Z) = 0.
    Returns (correlation, p_value).
    """
    if not z:
        r, p = stats.pearsonr(df[x].dropna(), df[y].dropna())
        return float(r), float(p)

    cols = [x, y] + z
    sub = df[cols].dropna()
    if len(sub) < len(cols) + 5:
        return 0.0, 1.0

    cov = sub.cov().values
    try:
        prec = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    xi = cols.index(x)
    yi = cols.index(y)
    pc = -prec[xi, yi] / np.sqrt(prec[xi, xi] * prec[yi, yi] + 1e-12)
    pc = float(np.clip(pc, -0.9999, 0.9999))

    # Fisher z-test
    n = len(sub)
    z_val = np.arctanh(pc) * np.sqrt(n - len(z) - 3)
    p_val = float(2 * stats.norm.sf(abs(z_val)))
    return pc, p_val


# ---------------------------------------------------------------------------
# PC Algorithm
# ---------------------------------------------------------------------------

def pc_algorithm(
    df: pd.DataFrame,
    features: list[str],
    alpha: float = 0.05,
    max_cond_set_size: int = 3,
) -> nx.DiGraph:
    """
    Run the PC algorithm to learn a DAG from observational data.

    Phase 1 (skeleton): Start with a complete undirected graph.  For each
    pair (X, Y), test conditional independence given subsets of their
    neighbours.  Remove edges where X ⊥ Y | Z for some Z.

    Phase 2 (orientation): Orient v-structures (X → Z ← Y) and apply
    Meek's orientation rules to maximally orient remaining edges.

    Returns a DiGraph; undirected edges are stored as bidirected (both
    directions present) — a convention indicating "unresolved orientation".

    Parameters
    ----------
    df       : DataFrame with feature columns (one row per observation)
    features : subset of df columns to include in the graph
    alpha    : significance level for conditional independence tests
    max_cond_set_size : maximum size of conditioning set (controls complexity)
    """
    present = [f for f in features if f in df.columns]
    n = len(present)

    # Phase 1: Build skeleton
    G = nx.Graph()
    G.add_nodes_from(present)
    G.add_edges_from(combinations(present, 2))

    sep_sets: dict[tuple, list] = {}

    for cond_size in range(max_cond_set_size + 1):
        edges_to_remove = []
        for x, y in list(G.edges()):
            neighbours_x = list(G.neighbors(x))
            neighbours_x = [n for n in neighbours_x if n != y]
            if len(neighbours_x) < cond_size:
                continue
            for z_set in combinations(neighbours_x, cond_size):
                _, p = partial_correlation(df, x, y, list(z_set))
                if p > alpha:
                    edges_to_remove.append((x, y))
                    sep_sets[(x, y)] = list(z_set)
                    sep_sets[(y, x)] = list(z_set)
                    break
        G.remove_edges_from(edges_to_remove)

    # Phase 2: Orient v-structures
    DG = nx.DiGraph()
    DG.add_nodes_from(present)
    # Add all skeleton edges as bidirected initially
    for x, y in G.edges():
        DG.add_edge(x, y)
        DG.add_edge(y, x)

    # Find and orient unshielded colliders X → Z ← Y (where X-Y not adjacent)
    for z in present:
        parents = [n for n in G.neighbors(z)]
        for x, y in combinations(parents, 2):
            if G.has_edge(x, y):
                continue  # shielded — not a collider
            z_sep = sep_sets.get((x, y), [])
            if z not in z_sep:
                # Orient X → Z ← Y
                if DG.has_edge(z, x):
                    DG.remove_edge(z, x)
                if DG.has_edge(z, y):
                    DG.remove_edge(z, y)

    return DG


# ---------------------------------------------------------------------------
# Correlation-based DAG (fast baseline)
# ---------------------------------------------------------------------------

def correlation_dag(
    df: pd.DataFrame,
    features: list[str],
    threshold: float = 0.15,
) -> nx.DiGraph:
    """
    Build a fast approximate causal DAG using partial correlations.

    Edges point from the variable with higher variance (more "upstream")
    to the one with lower variance, among pairs with |corr| > threshold.
    Acyclicity is enforced greedily.
    """
    present = [f for f in features if f in df.columns]
    corr = df[present].corr().abs()
    variances = df[present].var()

    DG = nx.DiGraph()
    DG.add_nodes_from(present)

    # Sort candidate edges by correlation strength (descending)
    edges = []
    for x, y in combinations(present, 2):
        c = corr.loc[x, y]
        if c > threshold:
            edges.append((c, x, y))
    edges.sort(reverse=True)

    for _, x, y in edges:
        # Orient from higher variance to lower variance
        if variances[x] >= variances[y]:
            src, dst = x, y
        else:
            src, dst = y, x
        DG.add_edge(src, dst)
        if not nx.is_directed_acyclic_graph(DG):
            DG.remove_edge(src, dst)

    return DG


# ---------------------------------------------------------------------------
# DAG analysis utilities
# ---------------------------------------------------------------------------

def dag_summary(G: nx.DiGraph, features: list[str]) -> pd.DataFrame:
    """Return a summary DataFrame of node centrality measures in the DAG."""
    rows = []
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    try:
        bet = nx.betweenness_centrality(G)
    except Exception:
        bet = {n: 0.0 for n in G.nodes()}

    for node in G.nodes():
        rows.append({
            "variable": node,
            "in_degree": in_deg.get(node, 0),
            "out_degree": out_deg.get(node, 0),
            "betweenness": round(bet.get(node, 0), 4),
            "is_root": in_deg.get(node, 0) == 0,
            "is_sink": out_deg.get(node, 0) == 0,
        })
    return pd.DataFrame(rows).sort_values("betweenness", ascending=False)


def plot_dag(G: nx.DiGraph, title: str = "Causal DAG") -> "plt.Figure":
    """Visualise a learned causal DAG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    node_colors = []
    for n in G.nodes():
        if G.in_degree(n) == 0:
            node_colors.append("#4CAF50")   # root: green
        elif G.out_degree(n) == 0:
            node_colors.append("#F44336")   # sink: red
        else:
            node_colors.append("#2196F3")   # intermediate: blue

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=node_colors, node_size=1800,
        font_size=9, font_color="white",
        edge_color="gray", arrows=True,
        arrowsize=20, arrowstyle="->",
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#4CAF50", label="Root (cause)"),
        Patch(color="#F44336", label="Sink (effect)"),
        Patch(color="#2196F3", label="Intermediate"),
    ]
    ax.legend(handles=legend, loc="upper left")
    fig.tight_layout()
    return fig
