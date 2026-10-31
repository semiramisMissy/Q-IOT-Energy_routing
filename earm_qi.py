"""
EARM-QI: Energy-Aware Routing with Quantum-Inspired Optimization
-----------------------------------------------------------------
پیاده‌سازی مرجع (نرم‌افزاری) برای مقاله.

نکته: این یک الگوریتم «کوانتوم‌الهام» است؛ به رایانه‌ی کوانتومی نیاز ندارد
و برتری آن بر روش‌های کلاسیک تضمین‌شده نیست و باید تجربی ارزیابی شود.

نیازمندی: matplotlib (برای رسم نمودارها).
"""

from __future__ import annotations
import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
import os
if not os.environ.get("DISPLAY") and os.name != "nt" and matplotlib.get_backend().lower() not in ("macosx",):
    matplotlib.use("Agg")          # بدون صفحه‌نمایش: فقط ذخیره‌ی فایل
import matplotlib.pyplot as plt

Path = Tuple[int, ...]


# ----------------------------------------------------------------------
# مدل شبکه
# ----------------------------------------------------------------------
@dataclass
class Node:
    x: float
    y: float
    e_init: float = 2.0          # انرژی اولیه (ژول)
    e_res: float = 2.0           # انرژی باقی‌مانده (ژول)


@dataclass
class Link:
    dist: float                  # فاصله (متر)
    delay: float                 # تأخیر لینک (میلی‌ثانیه)
    loss: float                  # احتمال افت بسته در [0,1)
    flap: float                  # نرخ ناپایداری/قطع‌ووصل شدن در [0,1)


class Network:
    def __init__(self) -> None:
        self.nodes: Dict[int, Node] = {}
        self.adj: Dict[int, Dict[int, Link]] = {}

    def add_node(self, nid: int, node: Node) -> None:
        self.nodes[nid] = node
        self.adj.setdefault(nid, {})

    def add_link(self, u: int, v: int, link: Link) -> None:
        self.adj[u][v] = link
        self.adj[v][u] = Link(link.dist, link.delay, link.loss, link.flap)


# ----------------------------------------------------------------------
# پارامترها
# ----------------------------------------------------------------------
@dataclass
class Params:
    # مدل انرژی رادیویی مرتبه‌ی اول
    e_elec: float = 50e-9        # J/bit
    eps_amp: float = 100e-12     # J/bit/m^2
    packet_bits: int = 4000

    # وزن‌های تابع هزینه (مجموع = 1)
    w_energy: float = 0.35
    w_delay: float = 0.25
    w_instab: float = 0.20
    w_loss: float = 0.20

    # تولید مسیر کاندید
    k_paths: int = 8

    # بخش کوانتوم‌الهام
    iterations: int = 60
    observations: int = 4        # تعداد اندازه‌گیری (نمونه‌برداری) در هر تکرار
    delta_theta: float = 0.08 * math.pi   # گام اولیه‌ی دروازه‌ی چرخش
    mutation_prob: float = 0.05  # جهش کوانتومی برای حفظ تنوع
    seed: Optional[int] = None

    # پایش دوره‌ای
    min_node_ratio: float = 0.10     # حداقل نسبت انرژی باقی‌مانده‌ی گره
    max_path_instab: float = 0.45    # آستانه‌ی ناپایداری کل مسیر
    max_path_loss: float = 0.50      # آستانه‌ی ریسک افت مسیر


@dataclass
class PathMetrics:
    energy: float
    delay: float
    instab: float
    loss: float


@dataclass
class RouteResult:
    path: Path
    cost: float
    metrics: PathMetrics
    probabilities: Dict[Path, float] = field(default_factory=dict)
    candidates: List[Path] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    history: List[List[float]] = field(default_factory=list)   # P_i در هر تکرار


# ----------------------------------------------------------------------
# گام ۲: تولید مسیرهای کاندید (Yen's K-shortest simple paths)
# ----------------------------------------------------------------------
def _dijkstra(adj, w, src, dst, banned_nodes, banned_edges):
    dist = {src: 0.0}
    prev: Dict[int, int] = {}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist.get(u, math.inf):
            continue
        for v in adj[u]:
            if v in banned_nodes or (u, v) in banned_edges:
                continue
            nd = d + w(u, v)
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return None
    p = [dst]
    while p[-1] != src:
        p.append(prev[p[-1]])
    return tuple(reversed(p))


def k_shortest_paths(net: Network, w, src: int, dst: int, k: int) -> List[Path]:
    first = _dijkstra(net.adj, w, src, dst, set(), set())
    if first is None:
        return []
    A: List[Path] = [first]
    B: List[Tuple[float, Path]] = []
    cost = lambda p: sum(w(p[i], p[i + 1]) for i in range(len(p) - 1))

    while len(A) < k:
        last = A[-1]
        for i in range(len(last) - 1):
            spur, root = last[i], last[: i + 1]
            banned_e = {(p[i], p[i + 1]) for p in A
                        if len(p) > i + 1 and p[: i + 1] == root}
            banned_n = set(root[:-1])
            spur_path = _dijkstra(net.adj, w, spur, dst, banned_n, banned_e)
            if spur_path:
                cand = root[:-1] + spur_path
                if cand not in A and all(cand != b for _, b in B):
                    heapq.heappush(B, (cost(cand), cand))
        if not B:
            break
        A.append(heapq.heappop(B)[1])
    return A


# ----------------------------------------------------------------------
# گام ۳: محاسبه‌ی معیارها برای هر مسیر
# ----------------------------------------------------------------------
def hop_energy(net: Network, u: int, v: int, p: Params) -> Tuple[float, float]:
    """انرژی ارسال در u و دریافت در v (ژول)."""
    d = net.adj[u][v].dist
    e_tx = p.packet_bits * (p.e_elec + p.eps_amp * d * d)
    e_rx = p.packet_bits * p.e_elec
    return e_tx, e_rx


def path_metrics(net: Network, path: Path, p: Params) -> PathMetrics:
    energy = delay = 0.0
    keep_instab = keep_ok = 1.0
    for u, v in zip(path[:-1], path[1:]):
        link = net.adj[u][v]
        e_tx, e_rx = hop_energy(net, u, v, p)
        # وزن‌دهی با نسبت انرژی باقی‌مانده: عبور از گره‌ی کم‌باتری گران‌تر است
        ru = max(net.nodes[u].e_res / net.nodes[u].e_init, 1e-3)
        rv = max(net.nodes[v].e_res / net.nodes[v].e_init, 1e-3)
        energy += e_tx / ru + e_rx / rv
        delay += link.delay
        keep_instab *= (1.0 - link.flap)
        keep_ok *= (1.0 - link.loss)
    return PathMetrics(energy, delay, 1.0 - keep_instab, 1.0 - keep_ok)


# ----------------------------------------------------------------------
# گام ۴ و ۵: نرمال‌سازی min-max و تابع هزینه
# ----------------------------------------------------------------------
def _minmax(vals: List[float]) -> List[float]:
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [0.0] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def path_costs(metrics: List[PathMetrics], p: Params) -> List[float]:
    e = _minmax([m.energy for m in metrics])
    d = _minmax([m.delay for m in metrics])
    s = _minmax([m.instab for m in metrics])
    l = _minmax([m.loss for m in metrics])
    return [p.w_energy * e[i] + p.w_delay * d[i] + p.w_instab * s[i] + p.w_loss * l[i]
            for i in range(len(metrics))]


# ----------------------------------------------------------------------
# گام ۶: به‌روزرسانی احتمال انتخاب با Q-bit و دروازه‌ی چرخش
# ----------------------------------------------------------------------
def quantum_inspired_search(costs: List[float], p: Params, rng: random.Random):
    """
    هر مسیر i یک Q-bit با زاویه‌ی θ_i دارد؛ احتمال انتخاب: P_i ∝ sin²(θ_i).
    در هر تکرار چند «اندازه‌گیری» انجام می‌شود (نمونه‌برداری از P)، سپس زاویه‌ی
    مسیر بهترینِ تاکنون به سمت بزرگ‌تر و بقیه به سمت کوچک‌تر می‌چرخند.
    """
    n = len(costs)
    theta = [math.pi / 4] * n                 # برهم‌نهی یکنواخت اولیه
    best_idx, best_cost = 0, math.inf
    c_lo, c_hi = min(costs), max(costs)
    span = max(c_hi - c_lo, 1e-12)

    def probs() -> List[float]:
        a = [math.sin(t) ** 2 for t in theta]
        z = sum(a)
        return [x / z for x in a]

    history: List[List[float]] = []
    for it in range(p.iterations):
        pr = probs()
        history.append(pr)
        # اندازه‌گیری
        for _ in range(p.observations):
            i = rng.choices(range(n), weights=pr)[0]
            if costs[i] < best_cost:
                best_idx, best_cost = i, costs[i]

        # چرخش: گام وابسته به فاصله‌ی هزینه تا بهترین (rotation gate تطبیقی)
        decay = 1.0 - it / p.iterations
        for i in range(n):
            gap = (costs[i] - best_cost) / span
            if i == best_idx:
                theta[i] += p.delta_theta * decay
            else:
                theta[i] -= p.delta_theta * decay * (0.3 + gap)
            # جهش کوانتومی (شبیه تونل‌زنی) برای فرار از بهینه‌ی محلی
            if rng.random() < p.mutation_prob:
                theta[i] += rng.uniform(-1, 1) * p.delta_theta
            theta[i] = min(max(theta[i], 0.05), math.pi / 2 - 0.05)

    return best_idx, probs(), history


# ----------------------------------------------------------------------
# الگوریتم اصلی EARM-QI
# ----------------------------------------------------------------------
class EARMQI:
    def __init__(self, net: Network, params: Optional[Params] = None) -> None:
        self.net = net
        self.p = params or Params()
        self.rng = random.Random(self.p.seed)
        self.current: Optional[RouteResult] = None
        self.src = self.dst = -1

    # وزن یال برای تولید کاندیدها (ترکیب سبک تأخیر، انرژی و افت)
    def _dead(self, n: int) -> bool:
        nd = self.net.nodes[n]
        return (n not in (self.src, self.dst)
                and nd.e_res / nd.e_init < self.p.min_node_ratio)

    def _gen_weight(self, u: int, v: int) -> float:
        # گره‌های تقریباً خالی از باتری از مسیرهای کاندید حذف می‌شوند
        if self._dead(u) or self._dead(v):
            return math.inf
        l = self.net.adj[u][v]
        e_tx, e_rx = hop_energy(self.net, u, v, self.p)
        ru = max(self.net.nodes[u].e_res / self.net.nodes[u].e_init, 1e-3)
        return (l.delay / 10.0) + (e_tx + e_rx) / ru * 1e3 + 5 * l.loss + 5 * l.flap

    def find_route(self, src: int, dst: int) -> Optional[RouteResult]:
        self.src, self.dst = src, dst
        # گام ۲
        cands = k_shortest_paths(self.net, self._gen_weight, src, dst, self.p.k_paths)
        if not cands:
            self.current = None
            return None
        # گام ۳–۵
        metrics = [path_metrics(self.net, c, self.p) for c in cands]
        costs = path_costs(metrics, self.p)
        # گام ۶
        _, probs, history = quantum_inspired_search(costs, self.p, self.rng)
        # گام ۷: انتخاب کمترین هزینه
        k = min(range(len(cands)), key=lambda i: costs[i])
        self.current = RouteResult(cands[k], costs[k], metrics[k],
                                   {cands[i]: probs[i] for i in range(len(cands))},
                                   cands, costs, history)
        return self.current

    # گام ۸: پایش دوره‌ای
    def path_is_healthy(self) -> Tuple[bool, str]:
        if self.current is None:
            return False, "no-route"
        path = self.current.path
        for n in path[1:-1]:
            nd = self.net.nodes[n]
            if nd.e_res / nd.e_init < self.p.min_node_ratio:
                return False, f"low-battery(node {n})"
        for u, v in zip(path[:-1], path[1:]):
            if v not in self.net.adj[u]:
                return False, f"link-down({u}-{v})"
        m = path_metrics(self.net, path, self.p)
        if m.instab > self.p.max_path_instab:
            return False, "instability"
        if m.loss > self.p.max_path_loss:
            return False, "packet-loss"
        return True, "ok"

    def maintain(self) -> Tuple[Optional[RouteResult], bool, str]:
        ok, reason = self.path_is_healthy()
        if ok:
            return self.current, False, reason
        return self.find_route(self.src, self.dst), True, reason


# ----------------------------------------------------------------------
# مثال اجرا: شبکه‌ی تصادفی + شبیه‌سازی مصرف انرژی و افت پایداری
# ----------------------------------------------------------------------
def build_random_network(n=40, area=100.0, radius=28.0, seed=1) -> Network:
    rng = random.Random(seed)
    net = Network()
    for i in range(n):
        net.add_node(i, Node(rng.uniform(0, area), rng.uniform(0, area),
                             e_init=2.0, e_res=rng.uniform(0.6, 2.0)))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = net.nodes[i], net.nodes[j]
            d = math.hypot(a.x - b.x, a.y - b.y)
            if d <= radius:
                net.add_link(i, j, Link(
                    dist=d,
                    delay=2 + d * 0.3 + rng.uniform(0, 3),
                    loss=min(0.4, 0.01 + d / radius * 0.1 + rng.uniform(0, 0.05)),
                    flap=min(0.4, rng.uniform(0.0, 0.12))))
    return net



# ----------------------------------------------------------------------
# رسم نمودارها
# ----------------------------------------------------------------------
OUT_DIR = "plots"


def _save(fig, name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150)


def plot_topology(net: Network, first: Path, last: Path, src: int, dst: int,
                  init_energy: Dict[int, float]) -> None:
    """توپولوژی؛ رنگ گره = انرژی باقی‌مانده‌ی نهایی، مسیر اولیه و نهایی هایلایت شده."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for u in net.adj:
        for v in net.adj[u]:
            if u < v:
                a, b = net.nodes[u], net.nodes[v]
                ax.plot([a.x, b.x], [a.y, b.y], color="lightgray", lw=0.6, zorder=1)

    def draw(path: Path, color: str, label: str, ls: str) -> None:
        xs = [net.nodes[n].x for n in path]
        ys = [net.nodes[n].y for n in path]
        ax.plot(xs, ys, color=color, lw=2.6, ls=ls, label=label, zorder=3)

    draw(first, "tab:blue", "initial route", "--")
    draw(last, "tab:red", "final route", "-")

    ids = list(net.nodes)
    ratios = [net.nodes[n].e_res / net.nodes[n].e_init for n in ids]
    sc = ax.scatter([net.nodes[n].x for n in ids], [net.nodes[n].y for n in ids],
                    c=ratios, cmap="RdYlGn", vmin=0, vmax=1, s=140,
                    edgecolors="k", linewidths=0.5, zorder=4)
    for n in ids:
        ax.text(net.nodes[n].x, net.nodes[n].y, str(n), fontsize=7,
                ha="center", va="center", zorder=5)
    for n, name in ((src, "SRC"), (dst, "DST")):
        ax.annotate(name, (net.nodes[n].x, net.nodes[n].y), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontweight="bold")
    fig.colorbar(sc, ax=ax, label="residual energy ratio (final)")
    ax.set_title("Network topology, initial vs final route")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="upper right"); ax.set_aspect("equal")
    _save(fig, "1_topology.png")


def plot_candidates(res: RouteResult) -> None:
    """هزینه‌ی هر مسیر کاندید و احتمال نهایی انتخاب آن."""
    n = len(res.candidates)
    labels = [f"P{i+1}\n({len(p)-1} hops)" for i, p in enumerate(res.candidates)]
    best = min(range(n), key=lambda i: res.costs[i])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["tab:red" if i == best else "tab:blue" for i in range(n)]
    a1.bar(labels, res.costs, color=colors)
    a1.set_title("Normalised cost of candidate paths (lower = better)")
    a1.set_ylabel("cost")
    probs = [res.probabilities[c] for c in res.candidates]
    a2.bar(labels, probs, color=colors)
    a2.set_title("Quantum-inspired selection probability")
    a2.set_ylabel("P(path)")
    _save(fig, "2_candidates.png")


def plot_convergence(res: RouteResult) -> None:
    """تغییر احتمال هر مسیر در طول تکرارهای جستجوی کوانتوم‌الهام."""
    hist = res.history
    n = len(res.candidates)
    best = min(range(n), key=lambda i: res.costs[i])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(n):
        ax.plot([h[i] for h in hist], lw=2.8 if i == best else 1.2,
                color="tab:red" if i == best else None,
                label=f"P{i+1}" + (" (best)" if i == best else ""))
    ax.set_xlabel("iteration"); ax.set_ylabel("selection probability")
    ax.set_title("Q-bit probability evolution")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, "3_convergence.png")


def plot_timeline(log: List[dict]) -> None:
    """رفتار مسیر فعال در طول دورهای شبیه‌سازی + نقاط محاسبه‌ی مجدد."""
    rounds = [r["round"] for r in log]
    reroutes = [r for r in log if r["changed"]]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    series = [("energy", "Energy metric (battery-weighted)", "tab:orange"),
              ("delay", "Delay (ms)", "tab:green"),
              ("instab", "Path instability", "tab:purple"),
              ("loss", "Packet-loss risk", "tab:red")]
    for ax, (key, title, col) in zip(axes, series):
        ax.plot(rounds, [r[key] for r in log], color=col, marker="o", ms=3)
        for r in reroutes:
            ax.axvline(r["round"], color="gray", ls=":", lw=0.8)
        ax.set_ylabel(title, fontsize=8)
    axes[2].axhline(log[0]["thr_instab"], color="k", ls="--", lw=0.8, label="threshold")
    axes[3].axhline(log[0]["thr_loss"], color="k", ls="--", lw=0.8, label="threshold")
    axes[2].legend(fontsize=8); axes[3].legend(fontsize=8)
    axes[0].set_title("Active route metrics over time (dotted lines = re-routing)")
    axes[-1].set_xlabel("round")
    _save(fig, "4_timeline_metrics.png")

    # انرژی شبکه و نمودار علت‌های محاسبه‌ی مجدد
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(rounds, [r["min_energy_ratio"] for r in log], label="min on active path", marker="o", ms=3)
    a1.plot(rounds, [r["mean_energy_ratio"] for r in log], label="network mean", marker="s", ms=3)
    a1.axhline(log[0]["thr_battery"], color="k", ls="--", lw=0.8, label="battery threshold")
    a1.set_xlabel("round"); a1.set_ylabel("residual energy ratio")
    a1.set_title("Residual energy"); a1.legend(fontsize=8)

    reasons: Dict[str, int] = {}
    for r in reroutes:
        key = r["reason"].split("(")[0]
        reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        a2.bar(list(reasons), list(reasons.values()), color="tab:gray")
    a2.set_title("Re-routing triggers"); a2.set_ylabel("count")
    _save(fig, "5_energy_and_triggers.png")


def main() -> None:
    net = build_random_network()
    algo = EARMQI(net, Params(seed=42))
    src, dst = 0, 39

    res = algo.find_route(src, dst)
    if res is None:
        print("مسیری یافت نشد.")
        return
    first_res = res
    print("مسیر اولیه:", res.path)
    print("هزینه: %.4f | %s" % (res.cost, res.metrics))

    rng = random.Random(7)
    log: List[dict] = []
    for rnd in range(1, 31):
        # مصرف انرژی بسته‌های ارسالی روی مسیر فعلی
        for u, v in zip(res.path[:-1], res.path[1:]):
            e_tx, e_rx = hop_energy(net, u, v, algo.p)
            net.nodes[u].e_res = max(0.0, net.nodes[u].e_res - e_tx * 150)
            net.nodes[v].e_res = max(0.0, net.nodes[v].e_res - e_rx * 150)
        # افت تصادفی کیفیت یکی از لینک‌های مسیر
        if rnd % 7 == 0:
            u, v = rng.choice(list(zip(res.path[:-1], res.path[1:])))
            net.adj[u][v].flap = net.adj[v][u].flap = 0.6
        new, changed, reason = algo.maintain()
        if new is None:
            print(f"دور {rnd}: مسیر موجود نیست ({reason})")
            break
        if changed:
            print(f"دور {rnd}: محاسبه‌ی مجدد ({reason}) -> {new.path}")
        res = new

        # ثبت وضعیت برای رسم نمودار
        m = path_metrics(net, res.path, algo.p)
        ratios = [net.nodes[n].e_res / net.nodes[n].e_init for n in res.path]
        log.append(dict(
            round=rnd, changed=changed, reason=reason, path=res.path,
            energy=m.energy, delay=m.delay, instab=m.instab, loss=m.loss,
            min_energy_ratio=min(ratios),
            mean_energy_ratio=sum(n.e_res / n.e_init for n in net.nodes.values()) / len(net.nodes),
            thr_instab=algo.p.max_path_instab, thr_loss=algo.p.max_path_loss,
            thr_battery=algo.p.min_node_ratio))

    # ---- نمودارها ----
    plot_topology(net, first_res.path, res.path, src, dst, {})
    plot_candidates(first_res)
    plot_convergence(first_res)
    if log:
        plot_timeline(log)
    print(f"نمودارها در پوشه‌ی '{OUT_DIR}/' ذخیره شدند.")
    plt.show()


if __name__ == "__main__":
    main()
