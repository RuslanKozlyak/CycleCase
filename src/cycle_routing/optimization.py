"""Derivative-free calibration of the edge-level comfort model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm

from .comfort import add_comfort_cost
from .config import TurnConfig
from .routing import EdgeKey, shortest_edge_path_with_turns


@dataclass(frozen=True, slots=True)
class ComfortParameter:
    """One positive nested comfort-config value and its search interval."""

    section: str
    key: str
    lower: float
    upper: float

    @property
    def name(self) -> str:
        return f"{self.section}.{self.key}"


DEFAULT_COMFORT_PARAMETERS = (
    ComfortParameter("highway", "cycleway", 0.40, 0.95),
    ComfortParameter("highway", "path", 0.55, 1.40),
    ComfortParameter("highway", "secondary", 1.00, 2.60),
    ComfortParameter("highway", "primary", 1.20, 3.60),
    ComfortParameter("surface", "unpaved", 1.00, 2.40),
    ComfortParameter("surface", "cobblestone", 1.00, 2.80),
    ComfortParameter("multipliers", "dedicated_cycleway", 0.45, 1.00),
    ComfortParameter("multipliers", "many_lanes", 1.00, 2.00),
)


@dataclass(frozen=True, slots=True)
class ComfortOptimizationConfig:
    """Controls the popularity objective and differential evolution."""

    seed: int = 42
    maxiter: int = 4
    popsize: int = 4
    polish: bool = False
    tolerance: float = 0.01
    free_detour_ratio: float = 0.10
    detour_penalty: float = 10.0
    coverage_reward: float = 0.50
    regularization: float = 0.05
    validation_candidates: int = 12
    show_progress: bool = True


@dataclass(slots=True)
class ComfortOptimizationResult:
    """Notebook-friendly optimization outputs."""

    optimized_config: dict[str, object]
    parameters: pd.DataFrame
    comparison: pd.DataFrame
    history: pd.DataFrame
    success: bool
    message: str
    nfev: int


def comfort_config_with_parameters(
    base_config: Mapping[str, object],
    values: Sequence[float],
    parameters: Sequence[ComfortParameter] = DEFAULT_COMFORT_PARAMETERS,
) -> dict[str, object]:
    """Return a deep copy of ``base_config`` with candidate values applied."""

    if len(values) != len(parameters):
        raise ValueError("The number of values must equal the number of parameters")
    result = deepcopy(base_config)
    for parameter, value in zip(parameters, values, strict=True):
        result[parameter.section][parameter.key] = float(value)
    return result


def spatial_od_split(
    graph,
    pairs,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> dict[str, list[tuple[str, int, int]]]:
    """Create deterministic west-to-east spatial train/validation/test blocks."""

    pairs = list(pairs)
    if len(pairs) < 5:
        raise ValueError("At least five OD pairs are required for a three-way split")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("The train and validation fractions must sum to less than one")

    ordered = sorted(
        pairs,
        key=lambda pair: (
            (float(graph.nodes[pair[1]]["x"]) + float(graph.nodes[pair[2]]["x"])) / 2,
            (float(graph.nodes[pair[1]]["y"]) + float(graph.nodes[pair[2]]["y"])) / 2,
            pair[0],
        ),
    )
    train_end = max(1, min(len(ordered) - 2, round(len(ordered) * train_fraction)))
    validation_size = max(1, round(len(ordered) * validation_fraction))
    validation_end = min(len(ordered) - 1, train_end + validation_size)
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def _edge_path_without_turns(graph, origin: int, destination: int, weight: str) -> list[EdgeKey]:
    node_path = nx.shortest_path(graph, origin, destination, weight=weight)
    edges: list[EdgeKey] = []
    for u, v in zip(node_path[:-1], node_path[1:], strict=True):
        key = min(
            graph[u][v],
            key=lambda candidate: float(
                graph.edges[u, v, candidate].get(
                    weight, graph.edges[u, v, candidate].get("length", 1.0)
                )
            ),
        )
        edges.append((u, v, key))
    return edges


def _comfort_edge_path(
    graph,
    origin: int,
    destination: int,
    turn_config: TurnConfig | None,
) -> list[EdgeKey]:
    if turn_config is not None and turn_config.enabled:
        result = shortest_edge_path_with_turns(
            graph,
            origin,
            destination,
            weight="comfort_cost",
            turn_config=turn_config,
        )
        if result is None:
            raise nx.NetworkXNoPath
        return result[0]
    return _edge_path_without_turns(graph, origin, destination, "comfort_cost")


def shortest_route_lengths(graph, pairs) -> dict[str, float]:
    """Cache physical shortest-path lengths for a fixed OD sample."""

    return {
        od_id: float(nx.shortest_path_length(graph, origin, destination, weight="length"))
        for od_id, origin, destination in pairs
    }


def score_comfort_config(
    graph,
    pairs,
    comfort_config: Mapping[str, object],
    *,
    shortest_lengths: Mapping[str, float] | None = None,
    optimization_config: ComfortOptimizationConfig | None = None,
    turn_config: TurnConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Score routes using matched traffic, coverage, and excess detour.

    Popularity is a length-weighted mean of ``log(1 + stad_trips)`` over the
    whole route. Unmatched edges therefore contribute zero. A small explicit
    coverage reward makes the behavior auditable, while detours beyond the free
    allowance are penalized quadratically.
    """

    settings = optimization_config or ComfortOptimizationConfig()
    pairs = list(pairs)
    shortest_lengths = dict(shortest_lengths or shortest_route_lengths(graph, pairs))
    add_comfort_cost(graph, comfort_config)
    rows = []
    for od_id, origin, destination in pairs:
        try:
            edges = _comfort_edge_path(graph, origin, destination, turn_config)
        except nx.NetworkXNoPath:
            rows.append(
                {
                    "od_id": od_id,
                    "route_length_m": np.nan,
                    "detour_ratio": np.nan,
                    "matched_length_share": 0.0,
                    "popularity_log_mean": 0.0,
                    "score": -1_000.0,
                }
            )
            continue

        lengths = np.asarray(
            [max(float(graph.edges[edge].get("length", 0.0)), 0.01) for edge in edges]
        )
        trips = np.asarray(
            [max(float(graph.edges[edge].get("stad_trips", 0.0)), 0.0) for edge in edges]
        )
        matched = np.asarray(
            [graph.edges[edge].get("stad_match_method", "none") != "none" for edge in edges]
        )
        route_length = float(lengths.sum())
        detour_ratio = route_length / shortest_lengths[od_id] - 1.0
        excess_detour = max(0.0, detour_ratio - settings.free_detour_ratio)
        popularity = float(np.average(np.log1p(trips), weights=lengths))
        coverage = float(lengths[matched].sum() / route_length)
        score = (
            popularity
            + settings.coverage_reward * coverage
            - settings.detour_penalty * excess_detour**2
        )
        rows.append(
            {
                "od_id": od_id,
                "route_length_m": route_length,
                "detour_ratio": detour_ratio,
                "matched_length_share": coverage,
                "popularity_log_mean": popularity,
                "score": score,
            }
        )

    details = pd.DataFrame(rows)
    summary = {
        "od_pairs": float(len(details)),
        "mean_score": float(details["score"].mean()),
        "median_popularity_log_mean": float(details["popularity_log_mean"].median()),
        "median_matched_length_share": float(details["matched_length_share"].median()),
        "median_detour_pct": float(100 * details["detour_ratio"].median()),
    }
    return details, summary


def optimize_comfort_weights(
    graph,
    splits: Mapping[str, Sequence[tuple[str, int, int]]],
    base_config: Mapping[str, object],
    *,
    parameters: Sequence[ComfortParameter] = DEFAULT_COMFORT_PARAMETERS,
    optimization_config: ComfortOptimizationConfig | None = None,
    turn_config: TurnConfig | None = None,
) -> ComfortOptimizationResult:
    """Fit comfort weights on ``train`` and report every supplied split."""

    settings = optimization_config or ComfortOptimizationConfig()
    train_pairs = list(splits.get("train", ()))
    if not train_pairs:
        raise ValueError("The 'train' split must contain at least one OD pair")
    all_pairs = [pair for split_pairs in splits.values() for pair in split_pairs]
    shortest_lengths = shortest_route_lengths(graph, all_pairs)
    base_values = np.asarray(
        [float(base_config[item.section][item.key]) for item in parameters], dtype=float
    )
    history: list[dict[str, float]] = []
    progress = tqdm(
        total=(settings.maxiter + 1) * settings.popsize * len(parameters),
        desc="Comfort weight search",
        disable=not settings.show_progress,
    )

    def objective(values: np.ndarray) -> float:
        candidate = comfort_config_with_parameters(base_config, values, parameters)
        _, summary = score_comfort_config(
            graph,
            train_pairs,
            candidate,
            shortest_lengths=shortest_lengths,
            optimization_config=settings,
            turn_config=turn_config,
        )
        regularization = float(np.mean(np.square(np.log(values / base_values))))
        loss = -summary["mean_score"] + settings.regularization * regularization
        history.append(
            {
                "evaluation": float(len(history) + 1),
                "loss": loss,
                "train_mean_score": summary["mean_score"],
                "regularization": regularization,
                **{item.name: float(value) for item, value in zip(parameters, values, strict=True)},
            }
        )
        progress.update(1)
        return loss

    try:
        result = differential_evolution(
            objective,
            bounds=[(item.lower, item.upper) for item in parameters],
            seed=settings.seed,
            maxiter=settings.maxiter,
            popsize=settings.popsize,
            polish=settings.polish,
            tol=settings.tolerance,
            workers=1,
            updating="immediate",
        )
    finally:
        progress.close()
    selected_values = np.asarray(result.x, dtype=float)
    validation_pairs = list(splits.get("validation", ()))
    if validation_pairs and settings.validation_candidates > 0:
        candidate_history = sorted(history, key=lambda row: row["loss"])
        best_validation_score = -np.inf
        seen_candidates: set[tuple[float, ...]] = set()
        checked = 0
        for row in tqdm(
            candidate_history,
            total=min(settings.validation_candidates, len(candidate_history)),
            desc="Validation model selection",
            disable=not settings.show_progress,
            leave=False,
        ):
            values = np.asarray([row[item.name] for item in parameters], dtype=float)
            identity = tuple(np.round(values, 12))
            if identity in seen_candidates:
                continue
            seen_candidates.add(identity)
            candidate = comfort_config_with_parameters(base_config, values, parameters)
            _, validation_summary = score_comfort_config(
                graph,
                validation_pairs,
                candidate,
                shortest_lengths=shortest_lengths,
                optimization_config=settings,
                turn_config=turn_config,
            )
            validation_score = (
                validation_summary["mean_score"]
                - settings.regularization * row["regularization"]
            )
            row["validation_mean_score"] = validation_summary["mean_score"]
            row["validation_selection_score"] = validation_score
            checked += 1
            if validation_score > best_validation_score:
                best_validation_score = validation_score
                selected_values = values
            if checked >= settings.validation_candidates:
                break

    optimized_config = comfort_config_with_parameters(
        base_config, selected_values, parameters
    )

    comparison_rows = []
    comparison_jobs = [
        (split_name, split_pairs, model, config)
        for split_name, split_pairs in splits.items()
        for model, config in (
            ("comfort_default", base_config),
            ("comfort_optimized", optimized_config),
        )
    ]
    for split_name, split_pairs, model, config in tqdm(
        comparison_jobs,
        desc="Default vs optimized",
        disable=not settings.show_progress,
        leave=False,
    ):
        _, summary = score_comfort_config(
            graph,
            split_pairs,
            config,
            shortest_lengths=shortest_lengths,
            optimization_config=settings,
            turn_config=turn_config,
        )
        comparison_rows.append({"split": split_name, "model": model, **summary})

    parameter_table = pd.DataFrame(
        {
            "parameter": [item.name for item in parameters],
            "lower": [item.lower for item in parameters],
            "current": base_values,
            "optimized": selected_values,
            "upper": [item.upper for item in parameters],
        }
    )
    add_comfort_cost(graph, optimized_config)
    return ComfortOptimizationResult(
        optimized_config=optimized_config,
        parameters=parameter_table,
        comparison=pd.DataFrame(comparison_rows),
        history=pd.DataFrame(history),
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
    )
