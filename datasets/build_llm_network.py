#!/usr/bin/env python3
"""
Build an LLM fine-tuning "contact network" for the Epidemic Model of Misalignment.

Two modes:
  1. --mode hf     Crawl the HuggingFace Hub API around a set of hub base models to
                   construct a REAL model-genealogy graph (edge: base_model -> fine-tune).
  2. --mode synth  Generate a synthetic scale-free (Barabasi-Albert) network calibrated
                   to empirical HF degree statistics (Llama-3-8B ~5200 children, etc.).

Output: an edgelist (parent<TAB>child) and a node metadata JSON, plus a GraphML file
        (requires networkx). Nodes carry: downloads, likes (real mode) used as a proxy
        for "connectivity / traffic" (hub-ness).

Reference topology anchors (Stalnaker et al. 2025, arXiv:2502.04484):
  meta-llama/Meta-Llama-3-8B      ~5200 fine-tuned children
  meta-llama/Llama-2-7b           ~3980
  mistralai/Mixtral-8x7B-Instruct ~3960
  meta-llama/Llama-2-7b-chat-hf   ~3670
This is a heavy-tailed / scale-free hub-and-spoke structure -> high <k^2> -> hubs
dominate the basic reproduction number R0 = (beta/gamma)(<k^2>-<k>)/<k> (Rozan 2025).
"""
import argparse, json, sys, time, urllib.request, urllib.parse
from collections import defaultdict

HUB_BASE_MODELS = [
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b",
]

def hf_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "epidemic-misalignment-research"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            print(f"  retry {attempt}: {e}", file=sys.stderr)
            time.sleep(3)
    return None

def children_of(base_model, limit=1000):
    """Return list of (id, downloads, likes) that declare base_model as base_model."""
    out = []
    q = urllib.parse.quote(base_model, safe="")
    url = (f"https://huggingface.co/api/models?filter=base_model:{q}"
           f"&limit={limit}&full=false&config=false&sort=downloads&direction=-1")
    data = hf_get(url)
    if not data:
        return out
    for m in data:
        out.append((m.get("id"), m.get("downloads", 0), m.get("likes", 0)))
    return out

def build_hf(depth=2, per_node_limit=400):
    edges = []
    meta = {}
    frontier = list(HUB_BASE_MODELS)
    seen = set(frontier)
    for m in frontier:
        meta.setdefault(m, {"downloads": None, "likes": None, "depth": 0})
    for d in range(depth):
        next_frontier = []
        for parent in frontier:
            kids = children_of(parent, limit=per_node_limit)
            print(f"[depth {d}] {parent}: {len(kids)} children", file=sys.stderr)
            for cid, dl, lk in kids:
                edges.append((parent, cid))
                if cid not in meta:
                    meta[cid] = {"downloads": dl, "likes": lk, "depth": d + 1}
                if cid not in seen:
                    seen.add(cid)
                    next_frontier.append(cid)
            time.sleep(0.3)
        frontier = next_frontier
        if not frontier:
            break
    return edges, meta

def build_synth(n=5000, m=2, seed=42):
    import networkx as nx
    G = nx.barabasi_albert_graph(n, m, seed=seed)  # scale-free, <k>=2m
    edges = [(f"node_{u}", f"node_{v}") for u, v in G.edges()]
    meta = {f"node_{i}": {"degree": G.degree(i), "depth": None} for i in G.nodes()}
    return edges, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hf", "synth"], default="hf")
    ap.add_argument("--out-prefix", default="datasets/llm_network")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--per-node-limit", type=int, default=400)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--m", type=int, default=2)
    args = ap.parse_args()

    if args.mode == "hf":
        edges, meta = build_hf(args.depth, args.per_node_limit)
    else:
        edges, meta = build_synth(args.n, args.m)

    el = f"{args.out_prefix}_{args.mode}_edgelist.tsv"
    mj = f"{args.out_prefix}_{args.mode}_nodes.json"
    with open(el, "w") as f:
        for a, b in edges:
            f.write(f"{a}\t{b}\n")
    with open(mj, "w") as f:
        json.dump(meta, f)
    # quick degree summary
    deg = defaultdict(int)
    for a, b in edges:
        deg[a] += 1; deg[b] += 1
    degs = sorted(deg.values(), reverse=True)
    n = len(deg); import statistics as st
    k = st.mean(degs) if degs else 0
    k2 = st.mean([d*d for d in degs]) if degs else 0
    print(json.dumps({
        "mode": args.mode, "nodes": n, "edges": len(edges),
        "mean_degree_<k>": round(k, 3), "<k^2>": round(k2, 3),
        "R0_factor_(<k^2>-<k>)/<k>": round((k2 - k)/k, 3) if k else None,
        "top10_degree": degs[:10],
        "edgelist": el, "nodes_json": mj,
    }, indent=2))

if __name__ == "__main__":
    main()
