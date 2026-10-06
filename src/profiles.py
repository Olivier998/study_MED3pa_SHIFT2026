"""Reading the MED3pa profiles (and their explanations) from the profile-tree HTML files."""

import json
import re
from pathlib import Path

import pandas as pd


def read_tree_nodes(html_path: Path) -> list:
    """Nodes of the `const treeData = [...]` array embedded in a MED3pa profile tree."""
    html = Path(html_path).read_text(encoding="utf-8")
    match = re.search(r"const\s+treeData\s*=\s*(\[.*?\]);", html, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No 'const treeData = ...;' found in {html_path}")
    return json.loads(match.group(1))


def summarize_node(node: dict) -> dict:
    """Profile path, metrics and explanation results (Detectron decision and
    p-value, FeatureResumer summary) of one tree node."""
    decision = p_value = feature_resumer = None
    for explanation in node.get("explanations", []):
        method = explanation.get("Method_name")
        result = explanation.get("result")
        if method == "Detectron" and isinstance(result, list) and result and isinstance(result[0], dict):
            decision = result[0].get("decision")
            p_value = result[0].get("p_value")
        elif method == "FeatureResumer":
            feature_resumer = result

    return {
        "node_id": node.get("node_id"),
        "path": json.dumps(node.get("path"), ensure_ascii=False),
        "metrics": json.dumps(node.get("metrics"), ensure_ascii=False),
        "node_information": json.dumps(node.get("node_information"), ensure_ascii=False),
        "decision": decision,
        "p_value": p_value,
        "FeatureResumer_result": feature_resumer,
    }


def parse_profile(path: list) -> list:
    """Parse a profile path such as ["*", "age <= 63.6053"] into a list of
    {"column", "operator", "value"} criteria ("*" is the whole population)."""
    conditions = []
    for item in path:
        if item == "*":
            continue
        match = re.match(r"(.+?)\s*(<=|>=|==|!=|<|>)\s*(.+)", item)
        if not match:
            raise ValueError(f"Cannot parse profile condition: {item}")
        column, operator, value = match.groups()
        conditions.append({"column": column.strip(), "operator": operator, "value": float(value)})
    return conditions


def load_profile_paths(csv_path: str) -> list:
    """Unique profile paths (in order of appearance) from the CSV written by
    experiments/extract_profiles.py."""
    paths = []
    for raw in pd.read_csv(csv_path)["path"]:
        path = json.loads(raw)
        if path not in paths:
            paths.append(path)
    return paths
