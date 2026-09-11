"""
probe_prompts.py
=================
Turns a build_probe_tasks.py item into (prompt_text, target_text) for
training, or (prompt_text, gold) for scoring. Shared by train_adapter.py
and eval_right_vs_wrong_graph.py so the two never drift apart.

The prompt NEVER contains node titles or any text derived from the
"New strategy"/"Strategy explanation" columns — only the bare list of
anonymized node ids present in the graph, so any correct answer must come
from the soft-prompt tokens, not from reading the question text.
"""
import json


def _id_legend(n_nodes: int) -> str:
    ids = ", ".join(f"N{i}" for i in range(n_nodes))
    return f"Node ids in this graph: {ids}."


def build_question(item: dict, n_nodes: int) -> str:
    task = item["task"]
    legend = _id_legend(n_nodes)

    if task == "adjacency":
        q = (f"{legend}\nUsing ONLY the graph structure encoded in the tokens above, "
             f"which node ids are DIRECTLY connected to {item['query_node']}? "
             f"Answer with a JSON list of node ids, e.g. [\"N1\", \"N4\"]. Use [] if none.")
    elif task == "node_type":
        q = (f"{legend}\nUsing ONLY the graph structure encoded in the tokens above, "
             f"what TYPE is node {item['query_node']}? Answer with exactly one of: "
             f"\"State\", \"Action\", \"Finding\", \"Unknown\".")
    elif task == "edge_type":
        a, b = item["query_edge"]
        q = (f"{legend}\nUsing ONLY the graph structure encoded in the tokens above, "
             f"there is an edge between {a} and {b}. What TYPE is it? Answer with exactly "
             f"one of: \"StateTransition\", \"SearchUpdate\", \"TrackUpdate\", \"Prediction\".")
    elif task == "two_hop":
        q = (f"{legend}\nUsing ONLY the graph structure encoded in the tokens above, "
             f"which node ids are reachable from {item['query_node']} in EXACTLY 2 hops? "
             f"Answer with a JSON list of node ids. Use [] if none.")
    elif task == "graph_aggregate":
        q = ("Using ONLY the graph structure encoded in the tokens above (no node list is "
             "given for this question), answer with a JSON object with exactly these keys: "
             "\"node_count_bucket\" (one of bucket_0..bucket_4), "
             "\"edge_count_bucket\" (one of bucket_0..bucket_4), "
             "\"density_bucket\" (one of bucket_0..bucket_4), "
             "\"dominant_node_type\" (one of \"State\", \"Action\", \"Finding\", \"Unknown\").")
    elif task == "graph_consistency":
        q = (f"{legend}\nHere is a claim about the graph encoded in the tokens above: "
             f"\"{item['claim']}\" Is this claim TRUE for the graph shown by the tokens above? "
             f"Answer with exactly one of: \"true\", \"false\".")
    else:
        raise ValueError(f"Unknown task: {task}")

    return (
        "You are given ONLY a graph representation as soft prompt tokens (no text "
        "description of the graph). Answer the question below using ONLY that "
        "representation. If you cannot tell, say \"unknown\" rather than guessing.\n\n"
        f"Question: {q}\n\n"
        "Respond with ONLY a JSON object: {\"answer\": ...}. No other text."
    )


def target_json(gold) -> str:
    return json.dumps({"answer": gold}, ensure_ascii=False)


def parse_answer(text: str):
    import re
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        try:
            return json.loads(m.group()).get("answer")
        except Exception:
            continue
    start, end = text.find("{"), text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end]).get("answer")
        except Exception:
            pass
    return None


def score(item: dict, pred) -> float:
    task = item["task"]
    gold = item["gold"]
    if pred is None:
        return 0.0

    if task in ("adjacency", "two_hop"):
        if not isinstance(pred, list):
            return 0.0
        p, g = set(map(str, pred)), set(gold)
        if not p and not g:
            return 1.0
        if not p or not g:
            return 0.0
        inter = len(p & g)
        prec, rec = inter / len(p), inter / len(g)
        return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

    if task in ("node_type", "edge_type"):
        return 1.0 if str(pred).strip().lower() == str(gold).strip().lower() else 0.0

    if task == "graph_consistency":
        pred_bool = str(pred).strip().lower() in ("true", "yes", "1")
        return 1.0 if pred_bool == bool(gold) else 0.0

    if task == "graph_aggregate":
        if not isinstance(pred, dict):
            return 0.0
        keys = ["node_count_bucket", "edge_count_bucket", "density_bucket", "dominant_node_type"]
        matched = sum(1 for k in keys
                      if str(pred.get(k, "")).strip().lower() == str(gold.get(k, "")).strip().lower())
        return matched / len(keys)

    raise ValueError(f"Unknown task: {task}")
