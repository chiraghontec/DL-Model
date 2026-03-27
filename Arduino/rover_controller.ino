/**
 * rover_controller.ino
 *
 * Unified firmware for the Frugal-Edge-AI Tomato Spraying Rover.
 *
 * Responsibilities:
 *   1. Z-axis linear actuator control (NEMA 17 via A4988/DRV8825)
 *      - AccelStepper for smooth acceleration / deceleration
 *      - End-stop homing (normally-closed micro-switch on Z_ENDSTOP_PIN)
 *   2. DC drive motors (L298N dual H-bridge)
 *      - Forward / backward / turn via Pi DRIVE commands  (reserved, future)
 *   3. Obstacle detection (HC-SR04 × 2, front + rear)
 *      - Polled every 100 ms; auto-stop if < OBSTACLE_THRESHOLD_CM
 *   4. Watchdog : auto-stop if no Pi heartbeat for WATCHDOG_MS ms
 *
 * Serial protocol (115200 baud, newline-terminated JSON):
 *
 *   Pi  → Arduino   {"cmd":"HOME"}
 *   Arduino → Pi    {"status":"HOMED"}
 *
 *   Pi  → Arduino   {"cmd":"GOTO","steps":4800}
 *   Arduino → Pi    {"status":"ARRIVED","steps":4800}
 *
 *   Pi  → Arduino   {"cmd":"STOP"}
 *   Arduino → Pi    {"status":"STOPPED","steps":<current>}
 *
 *   Pi  → Arduino   {"cmd":"POS"}
 *   Arduino → Pi    {"status":"POS","steps":<current>}
 *
 *   Pi  → Arduino   {"cmd":"PING"}
 *   Arduino → Pi    {"status":"PONG"}
 *
 *   Pi  → Arduino   {"cmd":"DRIVE","dir":"FWD","pwm":150}
 *   Arduino → Pi    {"status":"DRIVING","dir":"FWD"}
 *
 *   Pi  → Arduino   {"cmd":"HALT"}
 *   Arduino → Pi    {"status":"HALTED"}
 *
 * Pin assignments (see #defines below):
 *   Stepper  : STEP=D2, DIR=D3, EN=D4  (A4988 / DRV8825; EN active-LOW)
 *   End-stop : D5  (INPUT_PULLUP, NC switch → LOW when triggered)
 *   L298N    : IN1=D6, IN2=D7, ENA=D9(PWM), IN3=D10, IN4=D11, ENB=D12(PWM)
 *   HC-SR04 front : TRIG=A0, ECHO=A1
 *   HC-SR04 rear  : TRIG=A2, ECHO=A3
 *
 * Library required:
 *   AccelStepper ≥ 1.64  (Mike McCauley)
 *   ArduinoJson  ≥ 6.x   (Benoît Blanchon)
 *   Install via Tools → Manage Libraries in Arduino IDE.
 *
 * Hardware parameters (match config.json):
 *   NEMA 17 : 200 steps/rev, 1/16 microstepping → 3200 steps/rev
 *   GT2 2mm pitch, 20-tooth pulley  → 40 mm/rev
 *   steps_per_mm = 80
 *   travel_mm    = 800   → 64 000 steps total
 */

#include <AccelStepper.h>
#include <ArduinoJson.h>

// ─── Pin definitions ─────────────────────────────────────────────────────────
#define STEP_PIN          2
#define DIR_PIN           3
#define EN_PIN            4      // active-LOW enable

#define Z_ENDSTOP_PIN     5      // INPUT_PULLUP; LOW = triggered (NC)

// L298N drive motor pins
#define MOTOR_A_IN1       6
#define MOTOR_A_IN2       7
#define MOTOR_A_ENA       9     // PWM
#define MOTOR_B_IN3       10
#define MOTOR_B_IN4       11
#define MOTOR_B_ENB       12    // PWM (timer conflict — use analogWrite)

// HC-SR04 ultrasonic sensors
#define FRONT_TRIG        A0
#define FRONT_ECHO        A1
#define REAR_TRIG         A2
#define REAR_ECHO         A3

// ─── Constants ────────────────────────────────────────────────────────────────
#define MAX_SPEED          4000   // steps / sec
#define ACCELERATION       2000   // steps / sec²
#define MAX_STEPS          64000  // 800 mm × 80 steps/mm
#define HOMING_SPEED       -1200  // negative = toward end-stop
#define OBSTACLE_THRESHOLD_CM  15
#define OBSTACLE_POLL_MS   100
#define WATCHDOG_MS        3000   // stop if no Pi message in this period
#define SERIAL_BAUD        115200
#define JSON_BUF_SIZE      128

// ─── State ────────────────────────────────────────────────────────────────────
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

enum State { IDLE, HOMING, MOVING, STOPPED };
State state = IDLE;

long targetSteps  = 0;
bool homingActive = false;

unsigned long lastPingMs     = 0;
unsigned long lastObstacleMs = 0;

// ─── Helpers ─────────────────────────────────────────────────────────────────

void enableStepper(bool en) {
  // A4988/DRV8825 EN is active-LOW
  digitalWrite(EN_PIN, en ? LOW : HIGH);
}

long readUltrasonic(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  long duration = pulseIn(echoPin, HIGH, 30000UL); // 30 ms timeout
  if (duration == 0) return 999;                   // no echo → far/clear
  return duration / 58L;                           // cm
}

bool obstaclePresent() {
  long front = readUltrasonic(FRONT_TRIG, FRONT_ECHO);
  long rear  = readUltrasonic(REAR_TRIG,  REAR_ECHO);
  return (front < OBSTACLE_THRESHOLD_CM || rear < OBSTACLE_THRESHOLD_CM);
}

void driveStop() {
  digitalWrite(MOTOR_A_IN1, LOW);
  digitalWrite(MOTOR_A_IN2, LOW);
  analogWrite(MOTOR_A_ENA, 0);
  digitalWrite(MOTOR_B_IN3, LOW);
  digitalWrite(MOTOR_B_IN4, LOW);
  analogWrite(MOTOR_B_ENB, 0);
}

void driveForward(int pwm) {
  digitalWrite(MOTOR_A_IN1, HIGH);
  digitalWrite(MOTOR_A_IN2, LOW);
  analogWrite(MOTOR_A_ENA, pwm);
  digitalWrite(MOTOR_B_IN3, HIGH);
  digitalWrite(MOTOR_B_IN4, LOW);
  analogWrite(MOTOR_B_ENB, pwm);
}

void driveBackward(int pwm) {
  digitalWrite(MOTOR_A_IN1, LOW);
  digitalWrite(MOTOR_A_IN2, HIGH);
  analogWrite(MOTOR_A_ENA, pwm);
  digitalWrite(MOTOR_B_IN3, LOW);
  digitalWrite(MOTOR_B_IN4, HIGH);
  analogWrite(MOTOR_B_ENB, pwm);
}

void sendJson(const char* status, long stepsVal = -1) {
  StaticJsonDocument<JSON_BUF_SIZE> doc;
  doc["status"] = status;
  if (stepsVal >= 0) doc["steps"] = stepsVal;
  serializeJson(doc, Serial);
  Serial.println();
}

// ─── Command processor ────────────────────────────────────────────────────────

void processCommand(const char* buf) {
  StaticJsonDocument<JSON_BUF_SIZE> doc;
  DeserializationError err = deserializeJson(doc, buf);
  if (err) {
    sendJson("ERROR");
    return;
  }

  const char* cmd = doc["cmd"] | "";
  lastPingMs = millis();   // any message resets watchdog

  if (strcmp(cmd, "HOME") == 0) {
    // Drive toward end-stop at reduced speed
    enableStepper(true);
    stepper.setMaxSpeed(abs(HOMING_SPEED));
    stepper.setAcceleration(ACCELERATION);
    stepper.move(-MAX_STEPS * 2L);  // large negative move — stopped by endstop
    homingActive = true;
    state = HOMING;

  } else if (strcmp(cmd, "GOTO") == 0) {
    long steps = doc["steps"] | 0L;
    steps = constrain(steps, 0, MAX_STEPS);
    targetSteps = steps;
    enableStepper(true);
    stepper.setMaxSpeed(MAX_SPEED);
    stepper.setAcceleration(ACCELERATION);
    stepper.moveTo(steps);
    state = MOVING;

  } else if (strcmp(cmd, "STOP") == 0 || strcmp(cmd, "HALT") == 0) {
    stepper.stop();
    driveStop();
    enableStepper(false);
    state = STOPPED;
    const char* ackStatus = (strcmp(cmd, "HALT") == 0) ? "HALTED" : "STOPPED";
    sendJson(ackStatus, stepper.currentPosition());

  } else if (strcmp(cmd, "POS") == 0) {
    sendJson("POS", stepper.currentPosition());

  } else if (strcmp(cmd, "PING") == 0) {
    sendJson("PONG");

  } else if (strcmp(cmd, "DIST") == 0) {
    long front = readUltrasonic(FRONT_TRIG, FRONT_ECHO);
    long rear  = readUltrasonic(REAR_TRIG,  REAR_ECHO);
    StaticJsonDocument<JSON_BUF_SIZE> ack;
    ack["status"] = "DIST";
    ack["front_cm"] = front;
    ack["rear_cm"] = rear;
    serializeJson(ack, Serial);
    Serial.println();

  } else if (strcmp(cmd, "DRIVE") == 0) {
    const char* dir = doc["dir"] | "STOP";
    int pwm = doc["pwm"] | 150;
    pwm = constrain(pwm, 0, 255);
    if (strcmp(dir, "FWD") == 0)  driveForward(pwm);
    else if (strcmp(dir, "REV") == 0) driveBackward(pwm);
    else driveStop();
    // Ack
    StaticJsonDocument<JSON_BUF_SIZE> ack;
    ack["status"] = "DRIVING";
    ack["dir"] = dir;
    serializeJson(ack, Serial);
    Serial.println();

  } else {
    sendJson("ERROR");
  }
}

// ─── Setup ────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) {}   // wait for USB CDC on Leonardo/micro boards

  // Stepper pins
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN,  OUTPUT);
  pinMode(EN_PIN,   OUTPUT);
  enableStepper(false);    // start disabled until first command

  // End-stop
  pinMode(Z_ENDSTOP_PIN, INPUT_PULLUP);

  // Drive motor pins
  pinMode(MOTOR_A_IN1, OUTPUT);
  pinMode(MOTOR_A_IN2, OUTPUT);
  pinMode(MOTOR_A_ENA, OUTPUT);
  pinMode(MOTOR_B_IN3, OUTPUT);
  pinMode(MOTOR_B_IN4, OUTPUT);
  pinMode(MOTOR_B_ENB, OUTPUT);
  driveStop();

  // Ultrasonic
  pinMode(FRONT_TRIG, OUTPUT);
  pinMode(FRONT_ECHO, INPUT);
  pinMode(REAR_TRIG,  OUTPUT);
  pinMode(REAR_ECHO,  INPUT);

  stepper.setMaxSpeed(MAX_SPEED);
  stepper.setAcceleration(ACCELERATION);
  stepper.setCurrentPosition(0);

  lastPingMs = millis();
  sendJson("READY");
}

// ─── Main loop ────────────────────────────────────────────────────────────────

static char rxBuf[JSON_BUF_SIZE];
static uint8_t rxIdx = 0;

void loop() {
  unsigned long now = millis();

  // ── Read serial ─────────────────────────────────────────────────────────────
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (rxIdx > 0) {
        rxBuf[rxIdx] = '\0';
        processCommand(rxBuf);
        rxIdx = 0;
      }
    } else if (rxIdx < (JSON_BUF_SIZE - 1)) {
      rxBuf[rxIdx++] = c;
    }
  }

  // ── Homing state machine ────────────────────────────────────────────────────
  if (homingActive) {
    bool endstopHit = (digitalRead(Z_ENDSTOP_PIN) == LOW);
    if (endstopHit) {
      stepper.stop();
      stepper.setCurrentPosition(0);
      homingActive = false;
      enableStepper(false);
      state = IDLE;
      sendJson("HOMED", 0);
    } else {
      stepper.run();
    }
    return;   // skip rest of loop while homing
  }

  // ── Normal stepping ─────────────────────────────────────────────────────────
  if (state == MOVING) {
    stepper.run();
    if (stepper.distanceToGo() == 0) {
      enableStepper(false);
      state = IDLE;
      sendJson("ARRIVED", stepper.currentPosition());
    }
  }

  // ── Obstacle check ──────────────────────────────────────────────────────────
  if (now - lastObstacleMs >= OBSTACLE_POLL_MS) {
    lastObstacleMs = now;
    if (obstaclePresent()) {
      driveStop();
      // Do not stop the stepper Z-axis; only halt drive motors
    }
  }

  // ── Watchdog ─────────────────────────────────────────────────────────────────
  if (now - lastPingMs > WATCHDOG_MS) {
    driveStop();
    if (state == MOVING) {
      stepper.stop();
      enableStepper(false);
      state = STOPPED;
    }
    lastPingMs = now;   // reset to avoid spamming
  }
}
