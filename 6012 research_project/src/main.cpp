#include <Wire.h>
#include "PCA9685.h"

ServoDriver pwm;

const uint8_t ADDR = 0x70;

// Try 1 first (Seeed examples often use 1..16). If nothing moves, change to 0.
uint8_t SERVO_CH = 1;

void setup() {
  Wire.begin();
  Serial.begin(115200);

  pwm.init(ADDR);

  // If your servo range is off / buzzing, uncomment and tune:
  // pwm.setServoPulseRange(600, 2400, 180);

  Serial.println("Starting single-servo test...");
}

void loop() {
  pwm.setAngle(SERVO_CH, 70);  delay(800);
  pwm.setAngle(SERVO_CH, 110); delay(800);
}