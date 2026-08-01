"""Claim Chain 树代码本体（DevReady 资产 + DAG + 树遍历组装器）。

用户裁定(2026-07-27):仿真结构必须重构为 claim chain 树形式——模型 = 一棵 claim chain 树,
每个组成部分是带证据状态的命题;一切修改 = 树的有证据改写,禁补丁式试错。

架构:
  claim_nodes/*.yaml  = DevReady 资产(id/claim/state/freeze/evidence/refs/memory/guard/...)
  claim_dag.py        = ClaimNode + ClaimDAG(遍历/验证/修改检查)
  validate_modification(node_id, change)  = 修改前检查(validated 节点禁改)

状态机(用户裁定 2026-07-26):
  validated   → 已验证,禁改(红线的树解释)
  partial     → 部分验证,可动但需证据
  open        → 开放,可动但需归因
  falsified   → 已证伪,禁重走(方向错/双计)
  dead_end    → 死路留档,禁重走
"""
import os
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_NODES_DIR = os.path.join(_HERE, "claim_nodes")

STATE_RANK = {"validated": 4, "partial": 3, "open": 2, "falsified": 1, "dead_end": 0}


class ClaimNode:
    """一个命题节点(DevReady 资产实例)。"""

    def __init__(self, data: dict, path: str):
        self.id = data["id"]
        self.title = data["title"]
        self.claim = data["claim"].strip()
        self.state = data["state"]
        self.freeze = bool(data.get("freeze", False))
        self.evidence = list(data.get("evidence", []))
        self.refs = list(data.get("refs", []))
        self.memory = list(data.get("memory", []))
        self.guard = list(data.get("guard", []))
        self.falsified_when = data.get("falsified_when", "")
        self.depends_on = list(data.get("depends_on", []))
        self.provides_to = list(data.get("provides_to", []))
        self.children = [ClaimNode(c, path) for c in data.get("children", [])]
        self.version = data.get("version", "")
        self.changed = data.get("changed", "")
        self.notes = data.get("notes", "").strip()
        self._path = path

    @property
    def is_validated(self) -> bool:
        return self.state == "validated"

    @property
    def is_falsified(self) -> bool:
        return self.state in ("falsified", "dead_end")

    @property
    def is_modifiable(self) -> bool:
        return not self.freeze and self.state in ("partial", "open")

    def can_modify(self, change_desc: str) -> tuple:
        """修改前检查。返回 (允许?, 原因)。"""
        if self.freeze or self.state == "validated":
            return False, f"[{self.id}] {self.title}: validated 节点禁改(红线)"
        if self.state == "falsified":
            return False, f"[{self.id}] {self.title}: 已证伪节点,禁重走"
        if self.state == "dead_end":
            return False, f"[{self.id}] {self.title}: 死路留档,禁重走"
        return True, f"[{self.id}] {self.title}: {self.state} 可动(需证据+归因)"

    def transition(self, new_state: str, evidence_link: str, reason: str):
        """状态转移(必须带证据链接 + 原因)。"""
        if new_state == "validated" and self.state in ("falsified", "dead_end"):
            raise ValueError(f"[{self.id}] 已证伪节点不能转 validated,需开新节点")
        old = self.state
        self.state = new_state
        self.evidence.append(f"[{new_state}] {evidence_link} ({reason})")
        self.changed = f"{old} -> {new_state}: {reason}"
        return f"[{self.id}] {self.title}: {old} -> {new_state} ({reason})"

    def check_guard(self, metrics: dict) -> list:
        """检查守卫值是否越带。返回越带守卫列表。"""
        violations = []
        for g in self.guard:
            gid = g.get("id", "")
            metric = g.get("metric", "")
            band = g.get("band", "")
            rule = g.get("rule", "")
            # 简化:metrics 是 {guard_id: bool} (True=越带)
            if metrics.get(gid, False):
                violations.append({"id": gid, "metric": metric, "band": band, "rule": rule})
        return violations

    def __repr__(self):
        return f"<{self.id} {self.title} [{self.state}{'❄' if self.freeze else ''}]>"


class ClaimDAG:
    """Claim Chain 树(DAG)本体。"""

    def __init__(self, nodes_dir: str = None):
        self.nodes_dir = nodes_dir or _NODES_DIR
        self.nodes = {}
        self._load()

    def _load(self):
        for fn in sorted(os.listdir(self.nodes_dir)):
            if not fn.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(self.nodes_dir, fn)
            with open(path) as f:
                data = yaml.safe_load(f)
            node = ClaimNode(data, path)
            self.nodes[node.id] = node

    def get(self, node_id: str) -> ClaimNode:
        return self.nodes[node_id]

    def validate_modification(self, node_id: str, change_desc: str) -> tuple:
        """修改前检查:validated 节点禁改,已证伪节点禁重走。"""
        node = self.get(node_id)
        ok, reason = node.can_modify(change_desc)
        if not ok:
            return False, f"禁改: {reason}"
        return True, f"可动: {reason} — 需证据+归因(四步流程)"

    def check_all_guards(self, metrics: dict) -> dict:
        """检查所有节点守卫值。返回 {node_id: [violations]}。"""
        out = {}
        for nid, node in self.nodes.items():
            v = node.check_guard(metrics)
            if v:
                out[nid] = v
        return out

    def validated_nodes(self) -> list:
        return [n for n in self.nodes.values() if n.is_validated]

    def falsified_nodes(self) -> list:
        return [n for n in self.nodes.values() if n.is_falsified]

    def modifiable_nodes(self) -> list:
        return [n for n in self.nodes.values() if n.is_modifiable]

    def summary(self) -> str:
        lines = ["Claim Chain 树状态:"]
        for nid in sorted(self.nodes.keys()):
            n = self.nodes[nid]
            mark = "✓" if n.is_validated else ("✗" if n.is_falsified else ("~" if n.state == "partial" else "○"))
            lines.append(f"  {mark} [{nid}] {n.title} [{n.state}{'❄' if n.freeze else ''}]")
            for c in n.children:
                cmark = "✓" if c.is_validated else ("✗" if c.is_falsified else ("~" if c.state == "partial" else "○"))
                lines.append(f"    {cmark} [{c.id}] {c.title} [{c.state}{'❄' if c.freeze else ''}]")
        return "\n".join(lines)

    def export_tree_md(self) -> str:
        """导出 claim_tree.md 兼容格式。"""
        lines = ["# FLUXV RoboEagle 气动模型 — Claim Chain 树(代码本体生成)", ""]
        for nid in sorted(self.nodes.keys()):
            n = self.nodes[nid]
            mark = "✓" if n.is_validated else ("✗" if n.is_falsified else ("~" if n.state == "partial" else "○"))
            lines.append(f"{mark} [{nid}] {n.title} [{n.state}{'❄' if n.freeze else ''}]")
            lines.append(f"  claim: {n.claim[:100]}...")
            if n.children:
                for c in n.children:
                    cmark = "✓" if c.is_validated else ("✗" if c.is_falsified else ("~" if c.state == "partial" else "○"))
                    lines.append(f"  {cmark} [{c.id}] {c.title} [{c.state}{'❄' if c.freeze else ''}]")
        return "\n".join(lines)


# ---- 便捷接口(供 _v2_robo.py 或外部脚本调用) ----

_DAG = None

def get_dag() -> ClaimDAG:
    global _DAG
    if _DAG is None:
        _DAG = ClaimDAG()
    return _DAG

def validate_modification(node_id: str, change_desc: str) -> tuple:
    """修改前检查:validated 节点禁改。"""
    return get_dag().validate_modification(node_id, change_desc)

def check_guards(metrics: dict) -> dict:
    """检查所有节点守卫值。"""
    return get_dag().check_all_guards(metrics)


if __name__ == "__main__":
    dag = get_dag()
    print(dag.summary())
    print()
    print("修改前检查:")
    for nid in sorted(dag.nodes):
        ok, reason = dag.validate_modification(nid, "test change")
        print(f"  {nid}: {'✓' if ok else '✗'} {reason}")
