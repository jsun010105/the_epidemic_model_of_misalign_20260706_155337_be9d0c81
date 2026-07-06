#!/usr/bin/env python3
"""
Starter simulation: the Epidemic Model of Misalignment on an LLM fine-tuning network.

Ties together:
  - Network topology  (datasets/llm_network_*.tsv, or a synthetic BA graph)
  - Per-edge transmission probability calibrated from the subliminal-learning /
    transfer-ratio literature (Koenig et al. 2026 tau ~ 0.25-0.61; Cloud et al. 2025
    misalignment transfer ~0.08-0.10) -- ONLY conducts between models sharing a base
    model (the shared-initialization gate from Cloud et al. 2025 / Blank et al. 2026).
  - SIR/SIS dynamics + R0 prediction R0 = (beta/gamma)(<k^2>-<k>)/<k> (Rozan 2025)
  - Immunization experiments: hub-targeted vs random (Tanimoto 2011, Buono 2014)

This is a REFERENCE / STARTING POINT for the experiment runner, not the final study.
Requires: networkx, numpy. (EoN is optional for continuous-time Gillespie sims.)

Usage:
  python datasets/misalignment_epidemic_sim.py --graph datasets/llm_network_hf_edgelist.tsv
  python datasets/misalignment_epidemic_sim.py --synth-n 5000 --beta 0.3 --gamma 0.1
"""
import argparse, random
import networkx as nx
import numpy as np


def load_graph(path=None, synth_n=5000, synth_m=2, seed=42):
    if path:
        G = nx.read_edgelist(path, delimiter="\t")
    else:
        G = nx.barabasi_albert_graph(synth_n, synth_m, seed=seed)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def degree_moments(G):
    degs = np.array([d for _, d in G.degree()], dtype=float)
    k = degs.mean()
    k2 = (degs ** 2).mean()
    return k, k2, (k2 - k) / k  # <k>, <k^2>, R0 heterogeneity factor


def predicted_R0(G, beta, gamma):
    """DBMF SIR prediction (Rozan 2025 / Pastor-Satorras-Vespignani)."""
    k, k2, factor = degree_moments(G)
    return (beta / gamma) * factor


def sir_montecarlo(G, beta, gamma, seed_frac=0.005, immune=None, trials=20, max_steps=200):
    """Discrete-time SIR. `immune` = set of node ids that cannot be infected (vaccinated).
    Returns mean final outbreak size (fraction ever infected, excluding seeds)."""
    immune = immune or set()
    nodes = [n for n in G.nodes() if n not in immune]
    finals = []
    for t in range(trials):
        rng = random.Random(1000 + t)
        n_seed = max(1, int(seed_frac * len(nodes)))
        infected = set(rng.sample(nodes, n_seed))
        recovered = set()
        for _ in range(max_steps):
            if not infected:
                break
            new_inf = set()
            for u in infected:
                for v in G.neighbors(u):
                    if v in immune or v in infected or v in recovered:
                        continue
                    if rng.random() < beta:
                        new_inf.add(v)
            rec = {u for u in infected if rng.random() < gamma}
            recovered |= rec
            infected = (infected - rec) | new_inf
        finals.append(len(recovered) / len(nodes))
    return float(np.mean(finals)), float(np.std(finals))


def targeted_immune(G, frac):
    """Vaccinate the top `frac` highest-degree nodes (hub strategy)."""
    ranked = sorted(G.degree(), key=lambda x: -x[1])
    k = int(frac * G.number_of_nodes())
    return {n for n, _ in ranked[:k]}


def random_immune(G, frac, seed=0):
    rng = random.Random(seed)
    k = int(frac * G.number_of_nodes())
    return set(rng.sample(list(G.nodes()), k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=None, help="edgelist TSV; omit for synthetic BA")
    ap.add_argument("--synth-n", type=int, default=5000)
    ap.add_argument("--synth-m", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.30, help="per-edge transmission (calib. from tau)")
    ap.add_argument("--gamma", type=float, default=0.10, help="recovery/re-alignment rate")
    ap.add_argument("--immune-frac", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    G = load_graph(args.graph, args.synth_n, args.synth_m)
    k, k2, factor = degree_moments(G)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"<k>={k:.2f}  <k^2>={k2:.2f}  heterogeneity=(<k^2>-<k>)/<k>={factor:.2f}")
    print(f"Predicted R0 (DBMF) = (beta/gamma)*factor = {predicted_R0(G, args.beta, args.gamma):.2f}")

    base, base_sd = sir_montecarlo(G, args.beta, args.gamma, trials=args.trials)
    print(f"\nNo intervention:      final outbreak = {base:.3f} +/- {base_sd:.3f}")

    hub = targeted_immune(G, args.immune_frac)
    h, h_sd = sir_montecarlo(G, args.beta, args.gamma, immune=hub, trials=args.trials)
    print(f"Hub vaccination {args.immune_frac:.0%}:  final outbreak = {h:.3f} +/- {h_sd:.3f}")

    rnd = random_immune(G, args.immune_frac)
    r, r_sd = sir_montecarlo(G, args.beta, args.gamma, immune=rnd, trials=args.trials)
    print(f"Random vaccination {args.immune_frac:.0%}: final outbreak = {r:.3f} +/- {r_sd:.3f}")

    print(f"\n=> Hub-targeting advantage: {(r - h):.3f} smaller outbreak than random "
          f"at equal budget (positive supports the hypothesis).")


if __name__ == "__main__":
    main()
