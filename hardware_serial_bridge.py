"""
ML-Based Predictive TCP Congestion Control
Script: hardware_serial_bridge.py (PC-to-Microcontroller Hardware Bridge)

Team Members:
  - BAHAAR (Reg. No: 24BCE1760)
  - PARDHU (Reg. No: 24BCE1754)

Description:
  Streams real-time simulation telemetry to an external Arduino / ESP32
  hardware monitor via Serial (COM port) at 115200 baud.
  If no hardware microcontroller is currently plugged in, it enters
  a Hardware Emulation Mode to show how telemetry frames are transmitted.
"""

import time
import sys

def run_hardware_bridge(com_port="COM3", baud_rate=115200, mock=False):
    print("=" * 70)
    print("  ML-TCP HARDWARE TELEMETRY STREAMING BRIDGE")
    print("=" * 70)
    
    ser = None
    if not mock:
        try:
            import serial
            ser = serial.Serial(com_port, baud_rate, timeout=1)
            print(f"[+] Successfully connected to Hardware Node on {com_port} ({baud_rate} baud)")
            time.sleep(2)  # Wait for Arduino reset
        except Exception as e:
            print(f"[!] Physical Serial Port '{com_port}' not detected ({e}).")
            print("[*] Switching to HARDWARE TELEMETRY EMULATION MODE.")
            ser = None
    else:
        print("[*] Running in HARDWARE TELEMETRY EMULATION MODE.")
        
    print("\nStreaming live network packets to hardware display...")
    print("Press Ctrl+C to terminate stream.\n")
    
    queue = 0.0
    rate = 1.05
    base_rtt = 50.0
    prev_rtt = 50.0
    
    try:
        for step in range(1, 101):
            # Simulation tick
            rtt = base_rtt + (queue * 1.9)
            pred = 1 if queue >= 40.0 else 0
            
            # Rate control
            if pred == 1:
                rate = rate * 0.70  # Proactive throttle
            else:
                rate += 0.05
                
            queue += rate
            if queue > 50:
                queue = 50
                rate *= 0.558
            proc = min(queue, 3.0)
            queue -= proc
            
            # Construct Serial Telemetry Frame
            # Protocol: RTT:<val>,QUEUE:<val>,RATE:<val>,PRED:<0/1>\n
            frame = f"RTT:{rtt:.1f},QUEUE:{queue:.1f},RATE:{rate:.2f},PRED:{pred}\n"
            
            if ser is not None:
                ser.write(frame.encode('utf-8'))
            else:
                status_str = "[!] ML THROTTLE 30%" if pred == 1 else "[✓] NORMAL FLOW"
                print(f"Step {step:03d} -> Sent Hardware Frame: {frame.strip():<38} | {status_str}")
                
            time.sleep(0.08)  # 80ms transmission pacing
            
    except KeyboardInterrupt:
        print("\n[!] Stream stopped by user.")
    finally:
        if ser is not None:
            ser.close()
            print("[+] Serial port closed.")
            
    print("\n[Done] Hardware bridge finished.")

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "COM3"
    run_hardware_bridge(com_port=port, mock=True)
