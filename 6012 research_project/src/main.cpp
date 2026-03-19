#include <Wire.h>
#include "PCA9685.h"

ServoDriver servo;

const uint8_t PCA9685_ADDR = 0x7F;

// Change these to actual servo channels
// Example: cilium 1 uses channels 1 and 2, cilium 2 uses 3 and 4
struct Joint {
  uint8_t ch;
  float trimDeg;   // calibration offset if 90 is not truly upright
  bool invert;     // true if servo is mounted reversed
};

struct Cilium {
  Joint lower;
  Joint upper;
  float phaseOffset;   // 0.0 for now, later use e.g. 0.1, 0.2 etc.
};

Cilium cilia[2] = {
  { {1, -15.0f, false}, {2, 5.0f, false}, 0.1f },   // Cilium 1
  { {3, -10.0f, false}, {4, 5.0f, false}, 0.0f }    // Cilium 2
};

// Motion timing
const uint32_t CYCLE_MS  = 1800;   // total cycle time
const uint16_t UPDATE_MS = 20;     // servo update interval
uint32_t motionStartTime = 0;

// Motion angles (good safe starting point)
const float LOWER_BACK      = 75.0f;
const float LOWER_FORWARD   = 130.0f;

const float UPPER_UPRIGHT   = 90.0f;
const float UPPER_STRIKE    = 110.0f;
const float UPPER_FOLDED  = 165.0f;

// ----------------------------
// HELPER FUNCTIONS
// ----------------------------

float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

float lerp(float a, float b, float t) {
  return a + (b - a) * t;
}

// Smooth easing for nicer motion
float smoothstep(float t) {
  t = clampf(t, 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

float wrap01(float x) {
  while (x >= 1.0f) x -= 1.0f;
  while (x < 0.0f)  x += 1.0f;
  return x;
}

void gaitAtPhase(float phase, float &lowerDeg, float &upperDeg) {
  phase = wrap01(phase);

  // 1) STRIKE: lower moves forward, upper stays mostly upright
  if (phase < 0.40f) {
    float u = smoothstep(phase / 0.40f);
    lowerDeg = lerp(LOWER_BACK, LOWER_FORWARD, u);
    upperDeg = lerp(UPPER_UPRIGHT, UPPER_STRIKE, u);
  }

  // 2) WHIP/FOLD: upper quickly snaps to bent position
  else if (phase < 0.45f) {
    float u = smoothstep((phase - 0.40f) / 0.10f);
    lowerDeg = LOWER_FORWARD;
    upperDeg = lerp(UPPER_STRIKE, UPPER_FOLDED, u);
  }

  // 3) RECOVERY: lower comes back while upper stays folded
  else if (phase < 0.85f) {
    float u = smoothstep((phase - 0.50f) / 0.35f);
    lowerDeg = lerp(LOWER_FORWARD, LOWER_BACK, u);
    upperDeg = UPPER_FOLDED;
  }

  // 4) RESET: upper returns upright ready for next strike
  else {
    float u = smoothstep((phase - 0.85f) / 0.15f);
    lowerDeg = LOWER_BACK;
    upperDeg = lerp(UPPER_FOLDED, UPPER_UPRIGHT, u);
  }
}

void writeJoint(const Joint &j, float mechAngle) {
  float cmd = mechAngle;

  if (j.invert) {
    cmd = 180.0f - cmd;
  }

  cmd += j.trimDeg;
  cmd = clampf(cmd, 0.0f, 180.0f);

  servo.setAngle(j.ch, cmd);
}

void updateCilium(const Cilium &c, float masterPhase) {
  float lowerDeg, upperDeg;
  float localPhase = wrap01(masterPhase + c.phaseOffset);

  gaitAtPhase(localPhase, lowerDeg, upperDeg);

  writeJoint(c.lower, lowerDeg);
  writeJoint(c.upper, upperDeg);
}

void moveJointSmooth(const Joint &j, float startDeg, float endDeg, int steps, int stepDelayMs) {
  for (int k = 0; k <= steps; k++) {
    float t = (float)k / steps;
    float a = startDeg + (endDeg - startDeg) * t;
    writeJoint(j, a);
    delay(stepDelayMs);
  }
}

// ----------------------------
// ARDUINO SETUP / LOOP
// ----------------------------

void setup() {
  Wire.begin();
  Serial.begin(115200);

  servo.init(PCA9685_ADDR);
  servo.setServoPulseRange(600, 2400, 180);   // keep if your servos like it

  // Move both cilia gently into starting pose
for (int i = 0; i < 2; i++) {
  moveJointSmooth(cilia[i].lower, 90.0f, LOWER_BACK, 30, 20);
  moveJointSmooth(cilia[i].upper, 90.0f, UPPER_UPRIGHT, 30, 20);
  delay(1000);
  }
motionStartTime = millis();
}

void loop() {
  static uint32_t lastUpdate = 0;
  uint32_t now = millis();

  if (now - lastUpdate >= UPDATE_MS) {
    lastUpdate = now;

    uint32_t elapsed = now - motionStartTime;
    float masterPhase = (elapsed % CYCLE_MS) / (float)CYCLE_MS;

    // Both move simultaneously for now because both phaseOffset = 0
    for (int i = 0; i < 2; i++) {
      updateCilium(cilia[i], masterPhase);
    }
  }
}