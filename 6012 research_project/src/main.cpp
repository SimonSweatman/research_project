#include <Arduino.h>
#include <Wire.h>
#include <avr/pgmspace.h>
#include "PCA9685.h"
#include "gait_table.h"

ServoDriver servoBoard1;
ServoDriver servoBoard2;

// ----------------------------
// PCA9685 settings
// ----------------------------
const uint8_t PCA9685_ADDR_1 = 0x7F;  // board 1
const uint8_t PCA9685_ADDR_2 = 0x7E;  // board 2

// ----------------------------
// Servo / cilium definitions
// ----------------------------
struct Joint {
  uint8_t board;    // 0 = board 1, 1 = board 2
  uint8_t ch;       // channel on that board
  int8_t trimDeg;   // 90 -> 0 trim, 100 -> +10 trim, 80 -> -10 trim
};

struct Cilium {
  Joint lower;
  Joint upper;
};

// ----------------------------
// Cilia mapping
// ----------------------------
// Board 1: cilia 1-8, channels 1-16
// Board 2: cilia 9-14, channels 1-12
Cilium cilia[] = {
  { {0,  1,  10}, {0,  2, -10} },  // C1
  { {0,  3,  10}, {0,  4, -10} },  // C2
  { {0,  5,  10}, {0,  6,  10} },  // C3
  { {0,  7,  10}, {0,  8,   0} },  // C4
  { {0,  9,   0}, {0, 10,   0} },  // C5
  { {0, 11,  10}, {0, 12,   0} },  // C6
  { {0, 13,  10}, {0, 14,   0} },  // C7
  { {0, 15,  10}, {0, 16,   0} },  // C8

  { {1,  1,   0}, {1,  2,   0} },  // C9
  { {1,  3,  10}, {1,  4,   0} },  // C10
  { {1,  5,  10}, {1,  6,   5} },  // C11
  { {1,  7,  10}, {1,  8,   0} },  // C12
  { {1,  9,  10}, {1, 10,   5} },  // C13
  { {1, 11, -10}, {1, 12,  10} }   // C14
};

const uint8_t NUM_CILIA = sizeof(cilia) / sizeof(cilia[0]);

// ----------------------------
// Motion settings
// ----------------------------
const uint32_t CYCLE_MS = 4000;
const uint16_t UPDATE_MS = 20;
const uint16_t PHASE_STEP = 180;
const float SMOOTHING_ALPHA = 0.45f;

// ----------------------------
// Runtime state
// ----------------------------
uint32_t motionStartTime = 0;
uint32_t lastUpdateTime = 0;

float prevLower[NUM_CILIA];
float prevUpper[NUM_CILIA];
bool firstCommand = true;

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

float interpTableLower(float exactIndex) {
  while (exactIndex >= GAIT_TABLE_SIZE) exactIndex -= GAIT_TABLE_SIZE;
  while (exactIndex < 0) exactIndex += GAIT_TABLE_SIZE;

  uint16_t idx0 = (uint16_t)exactIndex;
  uint16_t idx1 = wrapIndex(idx0 + 1);
  float t = exactIndex - idx0;

  float a0 = tableLower(idx0);
  float a1 = tableLower(idx1);

  return a0 + (a1 - a0) * t;
}

float interpTableUpper(float exactIndex) {
  while (exactIndex >= GAIT_TABLE_SIZE) exactIndex -= GAIT_TABLE_SIZE;
  while (exactIndex < 0) exactIndex += GAIT_TABLE_SIZE;

  uint16_t idx0 = (uint16_t)exactIndex;
  uint16_t idx1 = wrapIndex(idx0 + 1);
  float t = exactIndex - idx0;

  float a0 = tableUpper(idx0);
  float a1 = tableUpper(idx1);

  return a0 + (a1 - a0) * t;
}

void writeJoint(const Joint &j, float cmdAngleDeg) {
  float cmd = cmdAngleDeg + j.trimDeg;

  cmd = clampf(cmd, 0.0f, 180.0f);

  if (j.board == 0) {
    servoBoard1.setAngle(j.ch, cmd);
  } else {
    servoBoard2.setAngle(j.ch, cmd);
  }
}

void writeCiliumFromTable(const Cilium &c, uint8_t ciliumIndex, float baseIndex) {
  // Reversed wave direction: C1 leads
  float phaseOffset = (NUM_CILIA - 1 - ciliumIndex) * PHASE_STEP;
  float exactIndex = baseIndex + phaseOffset;

  float lowerCmd = interpTableLower(exactIndex);
  float upperCmd = interpTableUpper(exactIndex);

  if (firstCommand) {
    prevLower[ciliumIndex] = lowerCmd;
    prevUpper[ciliumIndex] = upperCmd;
  } else {
    lowerCmd = prevLower[ciliumIndex] +
               SMOOTHING_ALPHA * (lowerCmd - prevLower[ciliumIndex]);

    upperCmd = prevUpper[ciliumIndex] +
               SMOOTHING_ALPHA * (upperCmd - prevUpper[ciliumIndex]);

    prevLower[ciliumIndex] = lowerCmd;
    prevUpper[ciliumIndex] = upperCmd;
  }

  writeJoint(c.lower, lowerCmd);
  writeJoint(c.upper, upperCmd);
}

void moveToStartPoseSmooth(uint16_t steps = 60, uint16_t stepDelayMs = 15) {
  const float START_LOWER = 90.0f;
  const float START_UPPER = 90.0f;

  for (uint16_t k = 0; k <= steps; k++) {
    float t = (float)k / (float)steps;

    for (uint8_t i = 0; i < NUM_CILIA; i++) {
      float phaseOffset = (NUM_CILIA - 1 - i) * PHASE_STEP;

      float targetLower = interpTableLower(phaseOffset);
      float targetUpper = interpTableUpper(phaseOffset);

      float lowerNow = START_LOWER + (targetLower - START_LOWER) * t;
      float upperNow = START_UPPER + (targetUpper - START_UPPER) * t;

      writeJoint(cilia[i].lower, lowerNow);
      writeJoint(cilia[i].upper, upperNow);

      prevLower[i] = targetLower;
      prevUpper[i] = targetUpper;
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

  servoBoard1.init(PCA9685_ADDR_1);
  servoBoard2.init(PCA9685_ADDR_2);

  servoBoard1.setServoPulseRange(600, 2400, 180);
  servoBoard2.setServoPulseRange(600, 2400, 180);

  moveToStartPoseSmooth();

  delay(300);

  motionStartTime = millis();
  lastUpdateTime = 0;
  firstCommand = true;
}

void loop() {
  uint32_t now = millis();

  if (now - lastUpdateTime < UPDATE_MS) {
    return;
  }

  lastUpdateTime = now;

  uint32_t elapsed = now - motionStartTime;

  float baseIndex =
    ((float)(elapsed % CYCLE_MS) * (float)GAIT_TABLE_SIZE) / (float)CYCLE_MS;

  for (uint8_t i = 0; i < NUM_CILIA; i++) {
    writeCiliumFromTable(cilia[i], i, baseIndex);
  }

  firstCommand = false;
}