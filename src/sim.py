"""
Unified Discrete-Event Physical Network Simulator
Supports fixed links, capacity walks, wireless fading (Rayleigh, Ricean, Markov Gilbert-Elliot),
buffer overflows, and multi-flow competition with strictly sender-observable telemetry.
Enforces strict physical packet conservation at every step:
sent = delivered + dropped + in_queue.
"""

import os

import numpy as np


def load_mahimahi_trace(trace_path="cellular_trace_mahimahi.log"):
    """Load or synthesize cellular trace for trace replay experiments."""
    if not os.path.exists(trace_path) and os.path.exists(
        os.path.join("results", trace_path)
    ):
        trace_path = os.path.join("results", trace_path)
    if os.path.exists(trace_path):
        with open(trace_path, "r", encoding="utf-8") as f:
            lines = [float(line.strip()) for line in f if line.strip()]
        if len(lines) > 1:
            intervals = np.diff(lines)
            cap_trace = 50.0 / np.clip(intervals, 5.0, 100.0)
            return cap_trace
    # Synthetic default fallback
    t = np.arange(3000)
    return np.clip(3.5 + 1.5 * np.sin(2 * np.pi * t / 400.0), 0.5, 12.0)


class SimResult(dict):
    """Result object supporting both dictionary and attribute-like access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def run_loss_sim(
    agent,
    total_steps=3000,
    seed=42,
    buffer_capacity=50.0,
    base_rtt=50.0,
    fixed_capacity=3.0,
    loss_mode="random",
    random_loss_rate=0.0,
    ge_p_good_to_bad=0.02,
    ge_p_bad_to_good=0.30,
    ge_bad_loss_prob=0.25,
):
    """
    Simulates single-flow traffic over a link with non-congestion (wireless) and congestion drops.
    Enforces strict packet conservation:
    sent_total = delivered_total + congestion_loss + wireless_loss + queue.
    """
    rng = np.random.RandomState(seed)
    queue = 0.0

    # Initialize agent without oracle link capacity knowledge
    agent.rate = 1.05
    agent.min_rtt = None
    agent.rtt_history = []
    agent.deliv_history = []
    agent.recent_losses = [0] * 6
    if hasattr(agent, "window"):
        agent.window = []
    if hasattr(agent, "cached_pred_cap"):
        agent.cached_pred_cap = 1.05
    if hasattr(agent, "step_cnt"):
        agent.step_cnt = 0
    if hasattr(agent, "standing_queue_steps"):
        agent.standing_queue_steps = 0
    if hasattr(agent, "in_competitive_mode"):
        agent.in_competitive_mode = False

    total_sent = 0.0
    total_delivered = 0.0
    congestion_loss = 0.0
    wireless_loss = 0.0
    rtt_log = []
    ge_state = 0

    for _ in range(total_steps):
        rate = max(0.02, agent.rate)
        total_sent += rate

        # 1. Non-congestion wireless loss check
        is_wireless_drop = False
        if loss_mode == "random":
            if random_loss_rate > 0 and rng.uniform() < random_loss_rate:
                is_wireless_drop = True
        elif loss_mode in ("gilbert_elliott", "burst"):
            if ge_state == 0 and rng.uniform() < ge_p_good_to_bad:
                ge_state = 1
            elif ge_state == 1 and rng.uniform() < ge_p_bad_to_good:
                ge_state = 0
            if ge_state == 1 and rng.uniform() < ge_bad_loss_prob:
                is_wireless_drop = True
        elif loss_mode == "fading":
            r_fade = np.sqrt(rng.normal(0, 1) ** 2 + rng.normal(0, 1) ** 2)
            if r_fade < 0.25:
                is_wireless_drop = True

        # 2. Ingress & buffer overflow check (congestion loss)
        is_congestion_drop = False
        queue += rate
        if queue > buffer_capacity:
            c_loss = queue - buffer_capacity
            congestion_loss += c_loss
            queue = buffer_capacity
            is_congestion_drop = True

        # 3. Router packet forwarding
        proc = min(queue, fixed_capacity)
        queue -= proc

        # 4. Wireless drop over the air (conserves packet count)
        w_drop = min(1.0, proc) if is_wireless_drop else 0.0
        proc_delivered = proc - w_drop
        wireless_loss += w_drop
        total_delivered += proc_delivered

        loss_event = is_congestion_drop or is_wireless_drop

        # 5. Realistic physical RTT telemetry
        curr_rtt = base_rtt + (queue * 1.9) + rng.normal(0, 0.8)
        curr_rtt = max(10.0, curr_rtt)
        rtt_log.append(curr_rtt)

        # 6. Agent observation and rate actuation
        agent.observe_ack(curr_rtt, loss_event, proc_delivered)
        agent.update_rate(loss_event)

    avg_rtt = float(np.mean(rtt_log))
    p95_rtt = float(np.percentile(rtt_log, 95))
    total_loss = congestion_loss + wireless_loss

    # Net application goodput (useful non-duplicate delivery)
    goodput = float((total_delivered - 20.0 - total_loss * 5.0) / total_steps)
    goodput = max(goodput, 0.001)

    conservation_error = abs(
        total_sent - (total_delivered + congestion_loss + wireless_loss + queue)
    )

    return SimResult(
        {
            "goodput": goodput,
            "throughput": float(total_delivered / total_steps),
            "avg_rtt": avg_rtt,
            "p95_rtt": p95_rtt,
            "total_sent": total_sent,
            "total_delivered": total_delivered,
            "total_loss": total_loss,
            "congestion_loss": congestion_loss,
            "wireless_loss": wireless_loss,
            "final_queue": queue,
            "conservation_error": conservation_error,
        }
    )


def run_capacity_sim(
    agent,
    family="markov",
    total_steps=3000,
    seed=42,
    buffer_capacity=50.0,
    base_rtt=50.0,
):
    """
    Simulates single-flow traffic over dynamic bandwidth channels:
    markov, mahimahi, step, or sinusoid.
    """
    rng = np.random.RandomState(seed)
    full_mahimahi = load_mahimahi_trace()

    # 1. Dynamic capacity trace generation
    cap_trace = np.zeros(total_steps)
    if family == "markov":
        state_caps = [1.2, 2.5, 5.0, 7.5]
        P = np.array(
            [
                [0.90, 0.08, 0.02, 0.00],
                [0.05, 0.88, 0.07, 0.00],
                [0.00, 0.06, 0.88, 0.06],
                [0.00, 0.02, 0.08, 0.90],
            ]
        )
        curr_state = int(rng.choice([1, 2]))
        for t in range(total_steps):
            if rng.uniform() < 0.10:
                curr_state = int(rng.choice(4, p=P[curr_state]))
            cap_trace[t] = state_caps[curr_state] + rng.normal(0, 0.08)
    elif family == "mahimahi":
        start_idx = int(rng.uniform(0, max(1, len(full_mahimahi) - total_steps - 50)))
        cap_trace = full_mahimahi[start_idx : start_idx + total_steps]
    elif family == "step":
        step_t = int(rng.uniform(800, 1500))
        for t in range(total_steps):
            cap_trace[t] = 6.0 if t < step_t else 1.8
            cap_trace[t] += rng.normal(0, 0.05)
    elif family == "sinusoid":
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.015, 0.025)
        for t in range(total_steps):
            cap_trace[t] = 3.0 + 1.5 * np.sin(freq * t + phase) + rng.normal(0, 0.05)
    cap_trace = np.clip(cap_trace, 0.5, 12.0)

    # 2. Simulation execution
    queue = 0.0
    agent.rate = 1.05
    agent.min_rtt = None
    agent.rtt_history = []
    agent.deliv_history = []
    agent.recent_losses = [0] * 6
    if hasattr(agent, "window"):
        agent.window = []
    if hasattr(agent, "cached_pred_cap"):
        agent.cached_pred_cap = 1.05
    if hasattr(agent, "step_cnt"):
        agent.step_cnt = 0
    if hasattr(agent, "standing_queue_steps"):
        agent.standing_queue_steps = 0
    if hasattr(agent, "in_competitive_mode"):
        agent.in_competitive_mode = False

    total_sent = 0.0
    total_delivered = 0.0
    congestion_loss = 0.0
    rtt_log = []

    for t in range(total_steps):
        rate = max(0.02, agent.rate)
        total_sent += rate
        queue += rate

        loss_event = False
        if queue > buffer_capacity:
            c_loss = queue - buffer_capacity
            congestion_loss += c_loss
            queue = buffer_capacity
            loss_event = True

        proc = min(queue, cap_trace[t])
        queue -= proc
        total_delivered += proc

        curr_rtt = base_rtt + (queue * 1.9) + rng.normal(0, 0.8)
        curr_rtt = max(10.0, curr_rtt)
        rtt_log.append(curr_rtt)

        agent.observe_ack(curr_rtt, loss_event, proc)
        agent.update_rate(loss_event)

    avg_rtt = float(np.mean(rtt_log))
    p95_rtt = float(np.percentile(rtt_log, 95))
    goodput = float((total_delivered - congestion_loss * 5.0) / total_steps)
    goodput = max(0.001, goodput)
    conservation_error = abs(total_sent - (total_delivered + congestion_loss + queue))

    return SimResult(
        {
            "goodput": goodput,
            "throughput": float(total_delivered / total_steps),
            "avg_rtt": avg_rtt,
            "p95_rtt": p95_rtt,
            "total_sent": total_sent,
            "total_delivered": total_delivered,
            "congestion_loss": congestion_loss,
            "final_queue": queue,
            "conservation_error": conservation_error,
        }
    )


def run_competition_sim(
    agent_host,
    agent_opp,
    total_steps=3000,
    seed=42,
    buffer_capacity=50.0,
    base_rtt=50.0,
    bottleneck_cap=6.0,
):
    """
    Simulates multi-flow competition between a host controller and an opponent cross-traffic controller
    over a shared bottleneck link.
    """
    rng = np.random.RandomState(seed)
    queue = 0.0

    agent_host.rate = 1.05
    agent_host.min_rtt = None
    agent_host.rtt_history = []
    agent_host.deliv_history = []
    agent_host.recent_losses = [0] * 6
    if hasattr(agent_host, "window"):
        agent_host.window = []
    if hasattr(agent_host, "standing_queue_steps"):
        agent_host.standing_queue_steps = 0
    if hasattr(agent_host, "in_competitive_mode"):
        agent_host.in_competitive_mode = False
    if hasattr(agent_host, "competitive_steps"):
        agent_host.competitive_steps = 0
    if hasattr(agent_host, "step_cnt"):
        agent_host.step_cnt = 0

    if agent_opp is not None:
        agent_opp.rate = 1.05
        agent_opp.min_rtt = None
        agent_opp.rtt_history = []
        agent_opp.deliv_history = []
        agent_opp.recent_losses = [0] * 6
        if hasattr(agent_opp, "window"):
            agent_opp.window = []
        if hasattr(agent_opp, "standing_queue_steps"):
            agent_opp.standing_queue_steps = 0
        if hasattr(agent_opp, "in_competitive_mode"):
            agent_opp.in_competitive_mode = False
        if hasattr(agent_opp, "step_cnt"):
            agent_opp.step_cnt = 0

    delivered_h, delivered_o = 0.0, 0.0
    sent_h, sent_o = 0.0, 0.0
    loss_host, loss_opp = 0.0, 0.0
    rtt_log_host = []

    for _ in range(total_steps):
        r_host = max(0.02, agent_host.rate)
        r_opp = max(0.02, agent_opp.rate) if agent_opp is not None else 0.0
        sent_h += r_host
        sent_o += r_opp

        queue += r_host + r_opp

        loss_h, loss_o = 0.0, 0.0
        if queue > buffer_capacity:
            excess = queue - buffer_capacity
            loss_h = excess * (r_host / max(r_host + r_opp, 1e-3))
            loss_o = excess * (r_opp / max(r_host + r_opp, 1e-3))
            loss_host += loss_h
            loss_opp += loss_o
            queue = buffer_capacity

        proc = min(queue, bottleneck_cap)
        queue -= proc

        proc_h = proc * (r_host / max(r_host + r_opp, 1e-3))
        proc_o = (
            proc * (r_opp / max(r_host + r_opp, 1e-3)) if agent_opp is not None else 0.0
        )

        delivered_h += proc_h
        delivered_o += proc_o

        curr_rtt = base_rtt + (queue * 1.9) + rng.normal(0, 0.8)
        curr_rtt = max(10.0, curr_rtt)
        rtt_log_host.append(curr_rtt)

        rtt_h = curr_rtt + rng.normal(0, 0.2)
        rtt_o = curr_rtt + rng.normal(0, 0.2)

        agent_host.observe_ack(rtt_h, (loss_h > 0), proc_h)
        agent_host.update_rate(loss_h > 0)

        if agent_opp is not None:
            agent_opp.observe_ack(rtt_o, (loss_o > 0), proc_o)
            agent_opp.update_rate(loss_o > 0)

    gp_h = float((delivered_h - loss_host * 5.0) / total_steps)
    gp_h = max(0.001, gp_h)

    if agent_opp is not None:
        gp_o = float((delivered_o - loss_opp * 5.0) / total_steps)
        gp_o = max(0.001, gp_o)
    else:
        gp_o = 0.0

    avg_rtt_h = float(np.mean(rtt_log_host))
    p95_rtt_h = float(np.percentile(rtt_log_host, 95))

    if agent_opp is not None:
        jfi = float(((gp_h + gp_o) ** 2) / (2.0 * (gp_h**2 + gp_o**2)))
        fair_share = float(gp_h / (gp_h + gp_o))
    else:
        jfi = 1.0
        fair_share = 1.0

    comp_steps = getattr(agent_host, "competitive_steps", 0)
    fpr_comp = float(comp_steps / total_steps)

    conservation_error = abs(
        (sent_h + sent_o) - (delivered_h + delivered_o + loss_host + loss_opp + queue)
    )

    return SimResult(
        {
            "gp_host": gp_h,
            "gp_opp": gp_o,
            "loss_host": loss_host,
            "loss_opp": loss_opp,
            "total_sent_host": sent_h,
            "total_sent_opp": sent_o,
            "total_delivered_host": delivered_h,
            "total_delivered_opp": delivered_o,
            "avg_rtt_host": avg_rtt_h,
            "p95_rtt_host": p95_rtt_h,
            "jfi": jfi,
            "fair_share": fair_share,
            "fpr_comp": fpr_comp,
            "conservation_error": conservation_error,
        }
    )


def run_multi_flow_sim(
    agents,
    total_steps=3000,
    seed=42,
    buffer_capacity=50.0,
    base_rtt=50.0,
    bottleneck_cap=12.0,
):
    """
    Simulates N competing flows sharing a bottleneck link.
    """
    rng = np.random.RandomState(seed)
    queue = 0.0
    n_flows = len(agents)

    # Initialize agents without oracle fair share
    for a in agents:
        a.rate = 1.05
        a.min_rtt = None
        a.rtt_history = []
        a.deliv_history = []
        a.recent_losses = [0] * 6
        if hasattr(a, "window"):
            a.window = []
        if hasattr(a, "step_cnt"):
            a.step_cnt = 0
        if hasattr(a, "standing_queue_steps"):
            a.standing_queue_steps = 0
        if hasattr(a, "in_competitive_mode"):
            a.in_competitive_mode = False

    delivered = np.zeros(n_flows)
    sent = np.zeros(n_flows)
    losses = np.zeros(n_flows)
    rtt_logs = [[] for _ in range(n_flows)]

    for _ in range(total_steps):
        rates = np.array([max(0.02, a.rate) for a in agents])
        sent += rates
        sum_rate = np.sum(rates)
        queue += sum_rate

        if queue > buffer_capacity:
            excess = queue - buffer_capacity
            flow_losses = excess * (rates / max(1e-5, sum_rate))
            losses += flow_losses
            queue = buffer_capacity
        else:
            flow_losses = np.zeros(n_flows)

        step_jitter = rng.normal(0.0, 0.05 * bottleneck_cap)
        step_cap = max(0.1, bottleneck_cap + step_jitter)
        proc = min(queue, step_cap)
        queue -= proc

        shares = proc * (rates / max(1e-5, sum_rate))
        delivered += shares

        curr_rtt = base_rtt + (queue * 1.9) + rng.normal(0, 0.8)
        curr_rtt = max(10.0, curr_rtt)

        for i in range(n_flows):
            rtt_i = curr_rtt + rng.normal(0, 0.2)
            rtt_logs[i].append(rtt_i)
            agents[i].observe_ack(rtt_i, flow_losses[i] > 0, shares[i])
            agents[i].update_rate(flow_losses[i] > 0)

    goodputs = [
        float(max(0.001, (delivered[i] - losses[i] * 5.0) / total_steps))
        for i in range(n_flows)
    ]
    all_rtts = [r for log in rtt_logs for r in log]
    avg_rtt = float(np.mean(all_rtts))
    p95_rtt = float(np.percentile(all_rtts, 95))
    total_loss = float(np.sum(losses))

    sum_gp = float(np.sum(goodputs))
    sum_sq_gp = float(np.sum(np.array(goodputs) ** 2))
    jfi = float((sum_gp**2) / (n_flows * sum_sq_gp)) if sum_sq_gp > 0 else 1.0
    conservation_error = abs(
        float(np.sum(sent)) - (float(np.sum(delivered)) + total_loss + queue)
    )

    return SimResult(
        {
            "goodputs": goodputs,
            "throughput": float(np.sum(delivered) / total_steps),
            "avg_rtt": avg_rtt,
            "p95_rtt": p95_rtt,
            "total_sent": float(np.sum(sent)),
            "total_delivered": float(np.sum(delivered)),
            "total_loss": total_loss,
            "final_queue": queue,
            "conservation_error": conservation_error,
            "jfi": jfi,
        }
    )
