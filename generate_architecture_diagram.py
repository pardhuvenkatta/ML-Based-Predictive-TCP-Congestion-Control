"""
Generates high-resolution architecture comparison diagram:
Legacy Cheated Architecture vs Audited Honest ML-TCP Architecture
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

def create_diagram(output_path="architecture_diagram.png"):
    fig = plt.figure(figsize=(16, 11), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Title
    ax.text(50, 97, "ML-Based Predictive TCP Congestion Control: Architectural Audit & Overhaul",
            fontsize=18, fontweight='bold', ha='center', va='center', color='#1a1a1a')
    ax.text(50, 93.8, "Prompt 1.5 Comparison: Flawed Ground-Truth Leakage vs. Audited RFC-Compliant End-to-End Control",
            fontsize=11.5, fontstyle='italic', ha='center', va='center', color='#4a5568')

    # ==========================================
    # LEFT PANEL: FLAWED / CHEATED ARCHITECTURE
    # ==========================================
    left_box = FancyBboxPatch((3, 35), 45, 54, boxstyle="round,pad=0.8,rounding_size=2",
                              edgecolor='#e53e3e', facecolor='#fff5f5', linewidth=2.5, linestyle='--')
    ax.add_patch(left_box)

    ax.text(25.5, 86, "[BEFORE] FLAWED CHEATED PIPELINE", fontsize=13, fontweight='bold', ha='center', color='#c53030')
    ax.text(25.5, 83, "(Unrealistic Information & Direct Buffer Spying)", fontsize=9.5, fontstyle='italic', ha='center', color='#742a2a')

    # Components Left: Sender
    s_left = FancyBboxPatch((6, 68), 39, 12, boxstyle="round,pad=0.5,rounding_size=1",
                            edgecolor='#c53030', facecolor='#ffffff', linewidth=1.5)
    ax.add_patch(s_left)
    ax.text(25.5, 77, "End-Host TCP Sender (Client/Server)", fontsize=11, fontweight='bold', ha='center', color='#2d3748')
    ax.text(25.5, 73.5, "• Reads Router Queue Size directly from buffer memory!", fontsize=9, ha='center', color='#e53e3e', fontweight='bold')
    ax.text(25.5, 70.5, "• Hardcoded Oracle Target: rate += (3.02 - rate) * 0.12", fontsize=9, ha='center', color='#e53e3e')

    # Left: ML Model Box
    ml_left = FancyBboxPatch((6, 52), 39, 12, boxstyle="round,pad=0.5,rounding_size=1",
                             edgecolor='#c53030', facecolor='#fed7d7', linewidth=1.5)
    ax.add_patch(ml_left)
    ax.text(25.5, 61, "Machine Learning Classifier (Random Forest)", fontsize=11, fontweight='bold', ha='center', color='#742a2a')
    ax.text(25.5, 57.5, "Input: [RTT, RTT_Change, Queue_Size, Sending_Rate]", fontsize=9, ha='center', color='#2d3748')
    ax.text(25.5, 54.5, "Target: Label = (Queue_Size > 40) at current step (Circular!)", fontsize=8.8, ha='center', color='#c53030', fontweight='bold')

    # Left: Router Box
    r_left = FancyBboxPatch((6, 37), 39, 11, boxstyle="round,pad=0.5,rounding_size=1",
                            edgecolor='#718096', facecolor='#edf2f7', linewidth=1.5)
    ax.add_patch(r_left)
    ax.text(25.5, 45, "Intermediate Router / Switch Buffer", fontsize=11, fontweight='bold', ha='center', color='#2d3748')
    ax.text(25.5, 41.5, "Buffer: 50 pkts | Service Rate: 3.0 pkts/step", fontsize=9, ha='center', color='#4a5568')
    ax.text(25.5, 38.8, "Cheating Bypasses: if queue >= 40: cut rate 30%", fontsize=8.8, ha='center', color='#e53e3e', fontweight='bold')

    # Red Arrows Left
    ax.annotate("", xy=(25.5, 68), xytext=(25.5, 64), arrowprops=dict(arrowstyle="->", color="#e53e3e", lw=2))
    ax.annotate("", xy=(25.5, 52), xytext=(25.5, 48), arrowprops=dict(arrowstyle="<->", color="#e53e3e", lw=2.5, ls='--'))
    ax.text(38, 50, "SPYING ON BUFFER\n(Illegal Leakage)", fontsize=8, color='#c53030', fontweight='bold', ha='center')

    # ==========================================
    # RIGHT PANEL: AUDITED HONEST ARCHITECTURE
    # ==========================================
    right_box = FancyBboxPatch((52, 35), 45, 54, boxstyle="round,pad=0.8,rounding_size=2",
                               edgecolor='#2f855a', facecolor='#f0fff4', linewidth=2.5)
    ax.add_patch(right_box)

    ax.text(74.5, 86, "[AFTER] AUDITED HONEST PIPELINE", fontsize=13, fontweight='bold', ha='center', color='#22543d')
    ax.text(74.5, 83, "(RFC 793/5681 Compliant End-to-End Observation)", fontsize=9.5, fontstyle='italic', ha='center', color='#276749')

    # Components Right: Sender
    s_right = FancyBboxPatch((55, 68), 39, 12, boxstyle="round,pad=0.5,rounding_size=1",
                             edgecolor='#2f855a', facecolor='#ffffff', linewidth=1.5)
    ax.add_patch(s_right)
    ax.text(74.5, 77, "End-Host TCP Sender (Pure Edge Observer)", fontsize=11, fontweight='bold', ha='center', color='#2d3748')
    ax.text(74.5, 73.5, "• 7 Sender-Only Signals: RTT ratio, Gradients (k=1,3,5)", fontsize=9, ha='center', color='#2f855a', fontweight='bold')
    ax.text(74.5, 70.5, "• Dynamic ACK Delivery Rate: EWMA (No hardcoded 3.02)", fontsize=9, ha='center', color='#2f855a')

    # Right: ML Model Box
    ml_right = FancyBboxPatch((55, 52), 39, 12, boxstyle="round,pad=0.5,rounding_size=1",
                              edgecolor='#2f855a', facecolor='#c6f6d5', linewidth=1.5)
    ax.add_patch(ml_right)
    ax.text(74.5, 61, "Future-Horizon Predictor (Random Forest)", fontsize=11, fontweight='bold', ha='center', color='#22543d')
    ax.text(74.5, 57.5, "Inputs: [rtt_ratio, grad_1, grad_3, grad_5, deliv_rate, send_rate, loss_5]", fontsize=8.2, ha='center', color='#1a202c')
    ax.text(74.5, 54.5, "Target: Future Drop in Next H=5 Steps (Zero Current Queue Input!)", fontsize=8.8, ha='center', color='#22543d', fontweight='bold')

    # Right: Router Box (Black Box)
    r_right = FancyBboxPatch((55, 37), 39, 11, boxstyle="round,pad=0.5,rounding_size=1",
                             edgecolor='#4a5568', facecolor='#edf2f7', linewidth=1.5)
    ax.add_patch(r_right)
    ax.text(74.5, 45, "Black-Box Network / Bottleneck Router", fontsize=11, fontweight='bold', ha='center', color='#2d3748')
    ax.text(74.5, 41.5, "Buffer State is PRIVATE & INVISIBLE to Host", fontsize=9, ha='center', color='#4a5568', fontweight='bold')
    ax.text(74.5, 38.8, "Feedback strictly through ACKs, RTT Delays & Loss Drops", fontsize=8.8, ha='center', color='#2b6cb0')

    # Green Arrows Right
    ax.annotate("", xy=(74.5, 68), xytext=(74.5, 64), arrowprops=dict(arrowstyle="->", color="#2f855a", lw=2))
    ax.annotate("", xy=(64, 48), xytext=(64, 68), arrowprops=dict(arrowstyle="->", color="#2b6cb0", lw=2))
    ax.text(60, 58, "Packets\n(Rate)", fontsize=8, color='#2b6cb0', fontweight='bold', ha='center')

    ax.annotate("", xy=(85, 68), xytext=(85, 48), arrowprops=dict(arrowstyle="->", color="#2f855a", lw=2))
    ax.text(89, 58, "ACK Feedback\n(RTT & Deliv)", fontsize=8, color='#2f855a', fontweight='bold', ha='center')

    # ==========================================
    # BOTTOM PANEL: CLOSED-LOOP INTERACTION CYCLE
    # ==========================================
    bot_box = FancyBboxPatch((3, 3), 94, 28, boxstyle="round,pad=0.8,rounding_size=2",
                             edgecolor='#3182ce', facecolor='#ebf8ff', linewidth=2.0)
    ax.add_patch(bot_box)

    ax.text(50, 28, "[CYCLE] AUDITED HONEST ML-TCP: CLOSED-LOOP STEP-BY-STEP EXECUTION CYCLE",
            fontsize=12.5, fontweight='bold', ha='center', color='#2b6cb0')

    steps = [
        ("Step 1: Telemetry Extraction",
         "Sample incoming ACK\nTrack min_RTT_seen\nCompute RTT ratio\nCompute Gradients (k=1,3,5)",
         9, '#2b6cb0'),
        ("Step 2: ML Model Inference",
         "7 Sender Features\nFast RF Tree Traversal\nEstimate Risk P(loss in H=5)\nInference Latency < 15 μs",
         32.5, '#2f855a'),
        ("Step 3: Dual-Mode Control",
         "If Risk >= 0.50: Proactive Cut\nIf Grad <= 0: Additive Inc (+0.05)\nElse: Pace to ACK Delivery Rate\n(No Hardcoded Capacity!)",
         56, '#d69e2e'),
        ("Step 4: Physical Evolution",
         "Queue += Sending_Rate\nProc = min(Queue, Link_Cap)\nQueue -= Proc\nDrop if Queue > Buffer",
         79.5, '#4a5568')
    ]

    for title, desc, x_pos, col in steps:
        step_patch = FancyBboxPatch((x_pos - 10, 6), 20, 19, boxstyle="round,pad=0.4,rounding_size=1",
                                    edgecolor=col, facecolor='#ffffff', linewidth=1.5)
        ax.add_patch(step_patch)
        ax.text(x_pos, 22.5, title, fontsize=9.2, fontweight='bold', ha='center', color=col)
        ax.text(x_pos, 13.5, desc, fontsize=8.2, ha='center', va='center', color='#2d3748')

    # Arrows between steps
    for i in range(3):
        x_from = steps[i][2] + 10
        x_to = steps[i+1][2] - 10
        ax.annotate("", xy=(x_to, 15.5), xytext=(x_from, 15.5),
                    arrowprops=dict(arrowstyle="->", color="#3182ce", lw=2.2))

    # Loop back arrow from Step 4 to Step 1
    ax.annotate("", xy=(9, 6), xytext=(79.5, 6),
                arrowprops=dict(arrowstyle="->", color="#3182ce", lw=1.8, connectionstyle="arc3,rad=-0.18", ls=':'))
    ax.text(45, 4.2, "Next Time Step Feedback (ACK Delivery & Observed RTT)", fontsize=8.5, fontstyle='italic', ha='center', color='#2b6cb0')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] High-resolution architecture diagram saved to '{output_path}'")

if __name__ == "__main__":
    create_diagram()
