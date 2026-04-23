#include <Arduino.h>
#include <Wire.h>
#include <avr/pgmspace.h>
#include "PCA9685.h"
#include "gait_table.h"

ServoDriver servo;

// ----------------------------
// PCA9685 settings
// ----------------------------
// Use the address that actually works on your board
const uint8_t PCA9685_ADDR = 0x7F;

// ----------------------------
// Servo / cilium definitions
// ----------------------------
struct Joint {
  uint8_t ch;       // PCA9685 channel
  int8_t trimDeg;   // calibration offset
  bool invert;      // reverse motion if mounted opposite way
};

struct Cilium {
  Joint lower;
  Joint upper;
};

// ----------------------------
// Cilia mapping
// ----------------------------
// Edit channel numbers, trims, and invert flags to match your real setup
Cilium cilia[] = {
  { {1, -15, false}, {2, 10, false} },  // cilium 1
  { {3, -10, false}, {4, 10, false} },  // cilium 2
  { {5, 5, false}, {6, 10, false} },   // cilium 3
  { {7, 10, false}, {8, 0, false} },   // cilium 4
  { {9, -10, false}, {10, 10, false} }   // cilium 5
};

const uint8_t NUM_CILIA = sizeof(cilia) / sizeof(cilia[0]);

// ----------------------------
// Motion settings
// ----------------------------
// One full gait cycle duration
const uint32_t CYCLE_MS = 1800;

// Servo update interval
const uint16_t UPDATE_MS = 20;

// Phase step between neighbouring cilia
// Since GAIT_TABLE_SIZE = 360, 20 steps = 20 degrees of phase
const uint16_t PHASE_STEP = 20;

// ----------------------------
// Runtime state
// ----------------------------
uint32_t motionStartTime = 0;
uint32_t lastUpdateTime = 0;

// ----------------------------
// Helpers
// ----------------------------
float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

uint16_t wrapIndex(int32_t idx) {
  while (idx >= (int32_t)GAIT_TABLE_SIZE) idx -= GAIT_TABLE_SIZE;
  while (idx < 0) idx += GAIT_TABLE_SIZE;
  return (uint16_t)idx;
}

uint8_t tableLower(uint16_t idx) {
  return pgm_read_byte(&LOWER_TABLE[idx]);
}

uint8_t tableUpper(uint16_t idx) {
  return pgm_read_byte(&UPPER_TABLE[idx]);
}

void writeJoint(const Joint &j, float cmdAngleDeg) {
  float cmd = cmdAngleDeg;

  if (j.invert) {
    cmd = 180.0f - cmd;
  }

  cmd += j.trimDeg;
  cmd = clampf(cmd, 0.0f, 180.0f);

  servo.setAngle(j.ch, cmd);
}

void writeCiliumFromTable(const Cilium &c, uint16_t baseIndex, uint16_t phaseOffsetSteps) {
  uint16_t idx = wrapIndex((int32_t)baseIndex + phaseOffsetSteps);

  uint8_t lowerCmd = tableLower(idx);
  uint8_t upperCmd = tableUpper(idx);

  writeJoint(c.lower, lowerCmd);
  writeJoint(c.upper, upperCmd);
}

void moveToStartPoseSmooth(uint16_t steps = 30, uint16_t stepDelayMs = 20) {
  const float START_LOWER = 90.0f;
  const float START_UPPER = 90.0f;

  const float targetLower = tableLower(0);
  const float targetUpper = tableUpper(0);

  for (uint16_t k = 0; k <= steps; k++) {
    float t = (float)k / (float)steps;

    float lowerNow = START_LOWER + (targetLower - START_LOWER) * t;
    float upperNow = START_UPPER + (targetUpper - START_UPPER) * t;

    for (uint8_t i = 0; i < NUM_CILIA; i++) {
      writeJoint(cilia[i].lower, lowerNow);
      writeJoint(cilia[i].upper, upperNow);
    }

    delay(stepDelayMs);
  }
}

// ----------------------------
// Arduino setup / loop
// ----------------------------
void setup() {
  Wire.begin();
  Serial.begin(115200);

  servo.init(PCA9685_ADDR);
  servo.setServoPulseRange(600, 2400, 180);

  // Smooth move into the phase-0 gait pose
  moveToStartPoseSmooth();

  delay(300);

  motionStartTime = millis();
  lastUpdateTime = 0;
}

void loop() {
  uint32_t now = millis();

  if (now - lastUpdateTime < UPDATE_MS) {
    return;
  }
  lastUpdateTime = now;

  uint32_t elapsed = now - motionStartTime;

  // Convert elapsed time into lookup index
  uint16_t baseIndex = (uint32_t)(elapsed % CYCLE_MS) * GAIT_TABLE_SIZE / CYCLE_MS;

  for (uint8_t i = 0; i < NUM_CILIA; i++) {
    // Reverse wave direction so cilium 1 leads and the wave propagates forward
    uint16_t phaseOffset = (NUM_CILIA - 1 - i) * PHASE_STEP;
    writeCiliumFromTable(cilia[i], baseIndex, phaseOffset);
  }
}