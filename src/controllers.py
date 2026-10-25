"""
Canonical Controllers Library for ML-TCP
Contains base protocol agents, traditional heuristics (Reno, CUBIC, Vegas, BBR, Copa),
non-ML pacing rules (Reference Non-ML D1, Default, Tuned, Multi-Condition, Ignore-All),
and ML-augmented agents (LossClassifier, CapacityForecaster, OpponentDetector).
"""

import numpy as np


class ProtocolAgent:
    """Base protocol agent providing sender-observable telemetry and EWMA rate tracking."""
    def __init__(self, name):
        self.name = name
        self.rate = 1.05
        self.min_rtt = None
        self.rtt_history = []
        self.recent_losses = [0] * 6
        self.est_delivery_rate = 1.0
        self.deliv_history = []
        self.throttle_count = 0
        self.step_cnt = 0
        self.standing_queue_steps = 0
        self.in_competitive_mode = False

    def get_min_rate(self):
        return max(0.02, 0.20 * self.est_delivery_rate)

    def observe_ack(self, rtt, loss_event, acked_proc):
        self.step_cnt += 1
        if self.min_rtt is None:
            self.min_rtt = rtt
            self.rtt_history = [rtt] * 6
            self.deliv_history = [acked_proc] * 20
        else:
            self.min_rtt = min(self.min_rtt, rtt)
            
        self.rtt_history.append(rtt)
        if len(self.rtt_history) > 20:
            self.rtt_history.pop(0)
            
        self.recent_losses.append(1 if loss_event else 0)
        if len(self.recent_losses) > 6:
            self.recent_losses.pop(0)
            
        self.deliv_history.append(acked_proc)
        if len(self.deliv_history) > 20:
            self.deliv_history.pop(0)
            
        self.est_delivery_rate = 0.85 * self.est_delivery_rate + 0.15 * acked_proc

    def update_rate(self, loss_event=False):
        pass


class PacingVariantD1_CoincidentRise(ProtocolAgent):
    """
    Reference Non-ML Controller (Variant D1: Coincident Rise Expiry).
    - 20-step rolling window to bridge wireless loss.
    - Flushes delivery history ONLY when queuing delay and gradient rise simultaneously
      (rtt_ratio > 1.15 and grad_1 > 0).
    - On packet loss WITHOUT delay inflation (rtt_ratio <= 1.12), delivery history is preserved.
    """
    def __init__(self, name):
        super().__init__(name)
        self.window = []

    def observe_ack(self, rtt, loss_event, acked_proc):
        super().observe_ack(rtt, loss_event, acked_proc)
        if not self.window:
            self.window = [acked_proc] * 20
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2] if len(self.rtt_history) >= 2 else 0.0
        if rtt_ratio > 1.15 and grad_1 > 0:
            self.window = [acked_proc]
        else:
            self.window.append(acked_proc)
            if len(self.window) > 20:
                self.window.pop(0)

    def update_rate(self, loss_event=False):
        min_rate = self.get_min_rate()
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        if loss_event:
            factor = 0.558 if rtt_ratio > 1.12 else 0.85
            self.rate = max(min_rate, self.rate * factor)
            return
        grad_1 = curr_rtt - self.rtt_history[-2] if len(self.rtt_history) >= 2 else 0.0
        target = max(self.window) if self.window else self.est_delivery_rate
        if grad_1 <= 0 and rtt_ratio < 1.12:
            self.rate += 0.05
        else:
            self.rate += (target - self.rate) * 0.14


class ReferenceNonMLAgent(PacingVariantD1_CoincidentRise):
    """Alias for Canonical Reference Non-ML Controller."""
    pass


class RenoAgent(ProtocolAgent):
    """Traditional AIMD (Additive Increase, Multiplicative Decrease)."""
    def __init__(self, name):
        super().__init__(name)
        self.cwnd = 2.0
        self.ssthresh = 64.0

    def update_rate(self, loss_event=False):
        min_rate = self.get_min_rate()
        if loss_event:
            self.ssthresh = max(2.0, self.cwnd / 2.0)
            self.cwnd = max(2.0, self.ssthresh)
        else:
            if self.cwnd < self.ssthresh:
                self.cwnd += 1.0  # Slow start
            else:
                self.cwnd += 1.0 / max(1.0, self.cwnd)  # Congestion avoidance
        curr_rtt = self.rtt_history[-1] if self.rtt_history else 50.0
        self.rate = max(min_rate, (self.cwnd / max(10.0, curr_rtt)) * 50.0)


class CubicAgent(ProtocolAgent):
    """Standard CUBIC TCP agent."""
    def __init__(self, name):
        super().__init__(name)
        self.cwnd = 2.0
        self.w_max = 2.0
        self.t_epoch = 0
        self.c = 0.4
        self.beta = 0.7

    def update_rate(self, loss_event=False):
        min_rate = self.get_min_rate()
        if loss_event:
            self.w_max = self.cwnd
            self.cwnd = max(2.0, self.cwnd * self.beta)
            self.t_epoch = 0
        else:
            self.t_epoch += 1
            t_sec = self.t_epoch * 0.05
            k = (self.w_max * (1.0 - self.beta) / self.c) ** (1.0 / 3.0)
            w_cubic = self.c * ((t_sec - k) ** 3) + self.w_max
            self.cwnd = max(2.0, w_cubic)
        curr_rtt = self.rtt_history[-1] if self.rtt_history else 50.0
        self.rate = max(min_rate, (self.cwnd / max(10.0, curr_rtt)) * 50.0)


class ReferenceVegasAgent(ProtocolAgent):
    """Delay-based TCP Vegas reference implementation."""
    def __init__(self, name, alpha=1.0, beta=3.0):
        super().__init__(name)
        self.alpha = alpha
        self.beta = beta

    def update_rate(self, loss_event=False):
        min_rate = self.get_min_rate()
        if loss_event:
            self.rate = max(min_rate, self.rate * 0.50)
            return
        curr_rtt = self.rtt_history[-1]
        expected_rate = self.rate * (self.min_rtt / max(curr_rtt, 1e-3))
        diff = self.rate - expected_rate
        if diff < self.alpha:
            self.rate += 0.05
        elif diff > self.beta:
            self.rate = max(min_rate, self.rate - 0.05)


class AuditedBBRAgent(ProtocolAgent):
    """
    Audited BBR agent with 10-RTT rolling BtlBw window and clamped cwnd headroom.
    """
    def __init__(self, name, window_rtts=10):
        super().__init__(name)
        self.state = "STARTUP"
        self.pacing_cycle = [1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.cycle_idx = 0
        self.btlbw = 1.0
        self.btlbw_window = []
        self.step_in_phase = 0

    def observe_ack(self, rtt, loss_event, acked_proc):
        super().observe_ack(rtt, loss_event, acked_proc)
        self.btlbw_window.append(acked_proc)
        if len(self.btlbw_window) > 20:
            self.btlbw_window.pop(0)
        self.btlbw = max(self.btlbw_window) if self.btlbw_window else 1.0

    def update_rate(self, loss_event=False):
        min_rate = self.get_min_rate()
        self.step_in_phase += 1
        if self.state == "STARTUP":
            self.rate = self.btlbw * 2.885
            if self.step_in_phase > 20:
                self.state = "DRAIN"
                self.step_in_phase = 0
        elif self.state == "DRAIN":
            self.rate = self.btlbw * (1.0 / 2.885)
            if self.step_in_phase > 15:
                self.state = "PROBE_BW"
                self.step_in_phase = 0
        elif self.state == "PROBE_BW":
            if self.step_in_phase % 8 == 0:
                self.cycle_idx = (self.cycle_idx + 1) % len(self.pacing_cycle)
            gain = self.pacing_cycle[self.cycle_idx]
            self.rate = max(min_rate, self.btlbw * gain)


class CopaAgent(ProtocolAgent):
    """Delay-based Copa agent (Arun & Balakrishnan, NSDI 2018)."""
    def __init__(self, name, delta=0.5):
        super().__init__(name)
        self.delta = delta

    def update_rate(self, loss_event=False):
        min_rate = self.get_min_rate()
        curr_rtt = self.rtt_history[-1]
        q_delay = max(0.001, (curr_rtt - self.min_rtt) / 1000.0)
        target_rate = 1.0 / (self.delta * q_delay)
        if self.rate < target_rate:
            self.rate += 0.05
        else:
            self.rate = max(min_rate, self.rate - 0.05)


class LossClassifierAgentBase(PacingVariantD1_CoincidentRise):
    """Loss classification controller with pluggable ML models and matched non-ML rules."""
    def __init__(self, name, mode="default_rule", threshold=1.12, cut_factor=0.90, model=None, multi_th=None):
        super().__init__(name)
        self.mode = mode
        self.threshold = threshold
        self.cut_factor = cut_factor
        self.model = model
        self.multi_th = multi_th

    def update_rate(self, loss_event=False):
        min_rate = self.get_min_rate()
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2] if len(self.rtt_history) >= 2 else 0.0
        grad_3 = (curr_rtt - self.rtt_history[-4]) / 3.0 if len(self.rtt_history) >= 4 else grad_1
        grad_5 = (curr_rtt - self.rtt_history[-6]) / 5.0 if len(self.rtt_history) >= 6 else grad_1
        deliv = max(self.window) if self.window else self.est_delivery_rate

        if loss_event:
            is_congestion = True
            if self.mode == "ignore_all":
                is_congestion = False
            elif self.mode == "model_off":
                is_congestion = True
            elif self.mode in ["default_rule", "tuned_rule"]:
                is_congestion = (rtt_ratio > self.threshold)
            elif self.mode == "multi_feature_rule":
                th_rtt, th_grad = self.multi_th if self.multi_th else (1.12, 0.5)
                is_congestion = (rtt_ratio > th_rtt and grad_1 > th_grad)
            elif self.mode in ["logreg", "tree", "rf"]:
                feats = np.array([[rtt_ratio, grad_1, grad_3, grad_5, deliv, self.rate, float(sum(self.recent_losses))]])
                is_congestion = bool(self.model.predict(feats)[0] == 1)

            factor = 0.558 if is_congestion else self.cut_factor
            self.rate = max(min_rate, self.rate * factor)
            return

        target = max(self.window) if self.window else self.est_delivery_rate
        if grad_1 <= 0 and rtt_ratio < 1.12:
            self.rate += 0.05
        else:
            self.rate += (target - self.rate) * 0.14


class CapacityAgentBase(ProtocolAgent):
    """Predictive capacity controller with LOFO models and matched heuristics."""
    def __init__(self, name, mode="default_nonml", model=None, flush_ratio=1.12, drain_factor=0.80):
        super().__init__(name)
        self.mode = mode
        self.model = model
        self.flush_ratio = flush_ratio
        self.drain_factor = drain_factor
        self.window = []
        self.ewma_cap = 3.0
        self.holt_level = 3.0
        self.holt_trend = 0.0
        self.cached_pred_cap = 3.0
        self.step_cnt = 0

    def observe_ack(self, rtt, loss_event, acked_proc):
        super().observe_ack(rtt, loss_event, acked_proc)
        if not self.window:
            self.window = [acked_proc] * 20
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2] if len(self.rtt_history) >= 2 else 0.0

        if rtt_ratio > self.flush_ratio and grad_1 > 0:
            self.window = [acked_proc]
        else:
            self.window.append(acked_proc)
            if len(self.window) > 20:
                self.window.pop(0)

        alpha = 0.20
        self.ewma_cap = alpha * acked_proc + (1.0 - alpha) * self.ewma_cap

        beta = 0.10
        prev_level = self.holt_level
        self.holt_level = alpha * acked_proc + (1.0 - alpha) * (self.holt_level + self.holt_trend)
        self.holt_trend = beta * (self.holt_level - prev_level) + (1.0 - beta) * self.holt_trend

    def update_rate(self, loss_event=False):
        min_rate = self.get_min_rate()
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2] if len(self.rtt_history) >= 2 else 0.0
        grad_3 = (curr_rtt - self.rtt_history[-4]) / 3.0 if len(self.rtt_history) >= 4 else grad_1
        grad_5 = (curr_rtt - self.rtt_history[-6]) / 5.0 if len(self.rtt_history) >= 6 else grad_1
        mean_deliv = float(np.mean(self.deliv_history)) if self.deliv_history else 1.0

        if self.step_cnt % 5 == 0:
            if self.mode == "persistence":
                self.cached_pred_cap = self.deliv_history[-1] if self.deliv_history else 3.0
            elif self.mode == "linear":
                slope = (self.deliv_history[-1] - self.deliv_history[-4]) / 3.0 if len(self.deliv_history) >= 4 else 0.0
                self.cached_pred_cap = max(0.5, self.deliv_history[-1] + slope * 5.0)
            elif self.mode == "ewma":
                self.cached_pred_cap = self.ewma_cap
            elif self.mode == "holt":
                self.cached_pred_cap = max(0.5, self.holt_level + 5.0 * self.holt_trend)
            elif self.mode in ["rf", "tree", "ridge"]:
                feats = np.array([[rtt_ratio, grad_1, grad_3, grad_5, mean_deliv, self.rate, float(sum(self.recent_losses))]])
                self.cached_pred_cap = float(self.model.predict(feats)[0])
            elif self.mode in ["default_nonml", "tuned_nonml", "model_off"]:
                self.cached_pred_cap = max(self.window) if self.window else self.est_delivery_rate

        if loss_event:
            factor = 0.558 if rtt_ratio > self.flush_ratio else self.drain_factor
            self.rate = max(min_rate, self.rate * factor)
            return

        pred_target = self.cached_pred_cap
        if self.mode in ["rf", "tree", "ridge", "persistence", "linear", "ewma", "holt"]:
            if rtt_ratio > 1.15:
                self.rate = max(min_rate, self.rate * 0.90)
            else:
                self.rate += (pred_target - self.rate) * 0.22
        else:
            target = max(self.window) if self.window else self.est_delivery_rate
            if grad_1 <= 0 and rtt_ratio < self.flush_ratio:
                self.rate += 0.05
            else:
                self.rate += (target - self.rate) * 0.14


class OpponentAgentBase(PacingVariantD1_CoincidentRise):
    """Opponent detection controller with dynamic competitive switching mode."""
    def __init__(self, name, mode="default_rule", threshold_steps=60, model=None):
        super().__init__(name)
        self.mode = mode
        self.threshold_steps = threshold_steps
        self.model = model
        self.detected_class = 0  # 0: Alone, 1: Loss, 2: Rate, 3: Delay
        self.competitive_steps = 0
        self.in_competitive_mode = False
        self.step_cnt = 0
        self.standing_queue_steps = 0

    def observe_ack(self, rtt, loss_event, acked_proc):
        super().observe_ack(rtt, loss_event, acked_proc)
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)

        if rtt_ratio > 1.12:
            self.standing_queue_steps += 1
        else:
            self.standing_queue_steps = max(0, self.standing_queue_steps - 2)

        if self.mode in ["rf", "lr", "tree"]:
            if self.step_cnt % 5 == 0 and self.step_cnt > 50:
                g1 = curr_rtt - self.rtt_history[-2] if len(self.rtt_history) >= 2 else 0.0
                g5 = (curr_rtt - self.rtt_history[-6]) / 5.0 if len(self.rtt_history) >= 6 else g1
                rate_ratio = self.rate / max(self.est_delivery_rate, 0.1)
                feats = np.array([[rtt_ratio, g1, g5, rate_ratio, float(self.standing_queue_steps)]])
                self.detected_class = int(self.model.predict(feats)[0])
            self.in_competitive_mode = (self.detected_class in [1, 2])
        elif self.mode == "default_rule":
            self.in_competitive_mode = (self.standing_queue_steps > 60)
        elif self.mode == "tuned_rule":
            self.in_competitive_mode = (self.standing_queue_steps > 78)
        elif self.mode == "multi_condition_rule":
            g5 = (curr_rtt - self.rtt_history[-6]) / 5.0 if len(self.rtt_history) >= 6 else 0.0
            self.in_competitive_mode = (self.standing_queue_steps > 65 and rtt_ratio > 1.15 and g5 >= 0)
        elif self.mode == "pure_pacing":
            self.in_competitive_mode = False

        if self.in_competitive_mode:
            self.competitive_steps += 1

    def update_rate(self, loss_event=False):
        min_rate = self.get_min_rate()
        curr_rtt = self.rtt_history[-1]
        rtt_ratio = curr_rtt / max(self.min_rtt, 1e-3)
        grad_1 = curr_rtt - self.rtt_history[-2] if len(self.rtt_history) >= 2 else 0.0

        if self.in_competitive_mode:
            if loss_event:
                self.rate = max(min_rate, self.rate * 0.70)
            else:
                self.rate += 0.10
            return

        if loss_event:
            factor = 0.558 if rtt_ratio > 1.12 else 0.85
            self.rate = max(min_rate, self.rate * factor)
            return

        target = max(self.window) if self.window else self.est_delivery_rate
        if grad_1 <= 0 and rtt_ratio < 1.12:
            self.rate += 0.05
        else:
            self.rate += (target - self.rate) * 0.14
