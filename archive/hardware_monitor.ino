/*
  ML-Based Predictive TCP Congestion Control
  Hardware Module: hardware_monitor.ino
  
  Team Members:
    - Student Research Team
    - Computer Networks Laboratory
    
  Target Hardware: ESP32 or Arduino Uno/Nano
  Peripherals:
    - 0.96" I2C OLED Display (SSD1306, 128x64)
    - 3 LEDs (Green = Pin 12, Yellow = Pin 14, Red = Pin 27)
    - Piezo Buzzer (Pin 26)
    
  Description:
    Receives real-time network telemetry over USB Serial (115200 baud)
    formatted as: "RTT:<val>,QUEUE:<val>,RATE:<val>,PRED:<0/1>"
    Visualizes router queue occupancy, RTT latency, and sounds alarm
    when ML predicts proactive congestion throttling (queue > 80%).
*/

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Hardware Pin Definitions (ESP32 standard pins)
const int PIN_LED_GREEN  = 12; // Normal (Queue < 70%)
const int PIN_LED_YELLOW = 14; // Warning (Queue 70% - 80%)
const int PIN_LED_RED    = 27; // Congestion / Throttle (Queue > 80%)
const int PIN_BUZZER     = 26; // Audio alert

void setup() {
  Serial.begin(115200);
  
  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_LED_YELLOW, OUTPUT);
  pinMode(PIN_LED_RED, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  
  digitalWrite(PIN_LED_GREEN, LOW);
  digitalWrite(PIN_LED_YELLOW, LOW);
  digitalWrite(PIN_LED_RED, LOW);
  digitalWrite(PIN_BUZZER, LOW);
  
  // Initialize I2C OLED (Address 0x3C is standard)
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    // If OLED not detected, continue Serial operation
  } else {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("ML-TCP Hardware Node");
    display.println("Bahaar & Pardhu");
    display.println("Initializing...");
    display.display();
    delay(1000);
  }
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    
    // Expected format: RTT:125.4,QUEUE:41.2,RATE:2.95,PRED:1
    if (line.startsWith("RTT:")) {
      float rtt = 0.0;
      float queue = 0.0;
      float rate = 0.0;
      int pred = 0;
      
      // Parse CSV telemetry tokens
      int idxQueue = line.indexOf("QUEUE:");
      int idxRate  = line.indexOf("RATE:");
      int idxPred  = line.indexOf("PRED:");
      
      if (idxQueue != -1 && idxRate != -1 && idxPred != -1) {
        rtt   = line.substring(4, idxQueue - 1).toFloat();
        queue = line.substring(idxQueue + 6, idxRate - 1).toFloat();
        rate  = line.substring(idxRate + 5, idxPred - 1).toFloat();
        pred  = line.substring(idxPred + 5).toInt();
        
        // Update Hardware LEDs & Buzzer
        if (pred == 1 || queue >= 40.0) {
          // Congestion Zone (>80% / 40 packets) -> RED LED + Chirp
          digitalWrite(PIN_LED_GREEN, LOW);
          digitalWrite(PIN_LED_YELLOW, LOW);
          digitalWrite(PIN_LED_RED, HIGH);
          tone(PIN_BUZZER, 1000, 40); // 40ms warning beep
        } else if (queue >= 35.0) {
          // Caution Zone (70% - 80%) -> YELLOW LED
          digitalWrite(PIN_LED_GREEN, LOW);
          digitalWrite(PIN_LED_YELLOW, HIGH);
          digitalWrite(PIN_LED_RED, LOW);
          noTone(PIN_BUZZER);
        } else {
          // Safe Zone (<70%) -> GREEN LED
          digitalWrite(PIN_LED_GREEN, HIGH);
          digitalWrite(PIN_LED_YELLOW, LOW);
          digitalWrite(PIN_LED_RED, LOW);
          noTone(PIN_BUZZER);
        }
        
        // Update OLED Graphical Display
        display.clearDisplay();
        display.setTextSize(1);
        display.setCursor(0, 0);
        display.print("ML-TCP ROUTER MONITOR");
        
        display.setCursor(0, 14);
        display.print("RTT:   "); display.print(rtt, 1); display.print(" ms");
        
        display.setCursor(0, 24);
        display.print("Rate:  "); display.print(rate, 2); display.print(" pkts/s");
        
        display.setCursor(0, 34);
        display.print("Queue: "); display.print(queue, 1); display.print(" / 50");
        
        // Draw Queue Progress Bar (0 to 50 packets mapped to 0 to 120 pixels)
        int barWidth = map((int)queue, 0, 50, 0, 120);
        display.drawRect(0, 46, 122, 8, SSD1306_WHITE);
        display.fillRect(1, 47, barWidth, 6, SSD1306_WHITE);
        
        // Status indicator
        display.setCursor(0, 56);
        if (pred == 1) {
          display.print("STATUS: ML THROTTLE 30%");
        } else {
          display.print("STATUS: OPTIMAL FLOW");
        }
        display.display();
      }
    }
  }
}
