/*
  ============================================================
   ESP32 Movement Controller — receives single-char commands
   from a Raspberry Pi over UART2, drives a 2-motor H-bridge
   (e.g. L298N / TB6612 / BTS7960 dual-channel style driver).
  ============================================================

  Wiring to the Raspberry Pi:
    RPi GPIO14 (TXD, physical pin 8)  -> ESP32 GPIO16 (RX2)
    RPi GPIO15 (RXD, physical pin 10) -> ESP32 GPIO17 (TX2)
    RPi GND                           -> ESP32 GND (common ground, required)

  Do NOT use GPIO1/GPIO3 (the ESP32's USB programming UART) for this link —
  you'd lose the ability to flash/debug over USB. UART2 on 16/17 keeps USB
  free.

  Baud rate matches the Python side (ESP32_BAUD_RATE = 115200).

  Command protocol (matches send_movement_command() in the RPi script):
    'w' -> drive forward
    's' -> drive backward
    'a' -> turn left in place
    'd' -> turn right in place
    'x' -> stop (all motors off)

  The RPi script already handles auto-stop timing on its side (it sends 'x'
  after a duration), so this firmware just needs to apply whatever command
  it last received and keep doing it until told otherwise — it does NOT
  need its own timers. That keeps the two sides from fighting over timing.

  >>> ADJUST THE MOTOR DRIVER PINS AND LOGIC BELOW to match your actual
  >>> driver board. This is written for a generic 2-motor H-bridge with
  >>> direction pins (IN1-IN4) + PWM enable pins (ENA/ENB), the most common
  >>> layout (L298N, TB6612FNG, many BTS7960 breakout boards).
*/

#include <Arduino.h>

// ── UART link to the Raspberry Pi ──────────────────────────
#define RPI_RX_PIN 16   // ESP32 RX2 <- RPi TX (GPIO14)
#define RPI_TX_PIN 17   // ESP32 TX2 -> RPi RX (GPIO15)
#define RPI_BAUD   115200
HardwareSerial RpiSerial(2);   // UART2

// ── Motor driver pins — CHANGE THESE to match your wiring ──
// Left motor
#define LEFT_IN1   26
#define LEFT_IN2   27
#define LEFT_EN    14   // PWM speed pin (ENA)

// Right motor
#define RIGHT_IN1  25
#define RIGHT_IN2  33
#define RIGHT_EN   32   // PWM speed pin (ENB)

// PWM config (ESP32 Arduino core 3.x supports analogWrite directly;
// if you're on core 2.x, swap analogWrite for ledcWrite with a channel).
#define MOTOR_SPEED 200   // 0-255. Tune to your motors/battery voltage.

char currentCommand = 'x';   // last command applied; 'x' = stopped

void setup() {
  Serial.begin(115200);          // USB serial, for debug prints only
  RpiSerial.begin(RPI_BAUD, SERIAL_8N1, RPI_RX_PIN, RPI_TX_PIN);

  pinMode(LEFT_IN1, OUTPUT);
  pinMode(LEFT_IN2, OUTPUT);
  pinMode(LEFT_EN, OUTPUT);
  pinMode(RIGHT_IN1, OUTPUT);
  pinMode(RIGHT_IN2, OUTPUT);
  pinMode(RIGHT_EN, OUTPUT);

  stopMotors();
  Serial.println("ESP32 movement controller ready — waiting on UART2 from RPi...");
}

void loop() {
  // Drain all bytes currently waiting; only the LAST byte in a burst
  // matters (older stale commands in the buffer are superseded).
  while (RpiSerial.available()) {
    char c = (char)RpiSerial.read();
    if (c == 'w' || c == 's' || c == 'a' || c == 'd' || c == 'x') {
      if (c != currentCommand) {
        currentCommand = c;
        applyCommand(currentCommand);
        Serial.print("Command applied: ");
        Serial.println(currentCommand);
      }
    }
    // Any other byte is ignored (protects against noise/garbage on the line).
  }
}

void applyCommand(char cmd) {
  switch (cmd) {
    case 'w': driveForward();  break;
    case 's': driveBackward(); break;
    case 'a': turnLeft();      break;
    case 'd': turnRight();     break;
    case 'x':
    default:  stopMotors();    break;
  }
}

// ── Low-level motor control ────────────────────────────────
// Adjust IN1/IN2 HIGH/LOW pairs if your motors spin the wrong direction —
// swapping the two wires for a motor is the standard fix, or swap the
// HIGH/LOW pair here instead.

void driveForward() {
  digitalWrite(LEFT_IN1, HIGH);  digitalWrite(LEFT_IN2, LOW);
  digitalWrite(RIGHT_IN1, HIGH); digitalWrite(RIGHT_IN2, LOW);
  analogWrite(LEFT_EN, MOTOR_SPEED);
  analogWrite(RIGHT_EN, MOTOR_SPEED);
}

void driveBackward() {
  digitalWrite(LEFT_IN1, LOW);  digitalWrite(LEFT_IN2, HIGH);
  digitalWrite(RIGHT_IN1, LOW); digitalWrite(RIGHT_IN2, HIGH);
  analogWrite(LEFT_EN, MOTOR_SPEED);
  analogWrite(RIGHT_EN, MOTOR_SPEED);
}

void turnLeft() {
  // Left wheels backward, right wheels forward -> rotate in place.
  digitalWrite(LEFT_IN1, LOW);  digitalWrite(LEFT_IN2, HIGH);
  digitalWrite(RIGHT_IN1, HIGH); digitalWrite(RIGHT_IN2, LOW);
  analogWrite(LEFT_EN, MOTOR_SPEED);
  analogWrite(RIGHT_EN, MOTOR_SPEED);
}

void turnRight() {
  // Right wheels backward, left wheels forward -> rotate in place.
  digitalWrite(LEFT_IN1, HIGH); digitalWrite(LEFT_IN2, LOW);
  digitalWrite(RIGHT_IN1, LOW); digitalWrite(RIGHT_IN2, HIGH);
  analogWrite(LEFT_EN, MOTOR_SPEED);
  analogWrite(RIGHT_EN, MOTOR_SPEED);
}

void stopMotors() {
  digitalWrite(LEFT_IN1, LOW);  digitalWrite(LEFT_IN2, LOW);
  digitalWrite(RIGHT_IN1, LOW); digitalWrite(RIGHT_IN2, LOW);
  analogWrite(LEFT_EN, 0);
  analogWrite(RIGHT_EN, 0);
}
