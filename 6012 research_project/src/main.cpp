#include <Arduino.h>
#include <Wire.h>

#include "PCA9685.h"
#include "gait_table.h"

ServoDriver servo;

// ============================================================
// HARDWARE SETTINGS
// ============================================================

constexpr uint8_t PCA_ADDR = 0x7E;

// Active cilium
constexpr uint8_t ACTIVE_LOWER_CH = 11;
constexpr uint8_t ACTIVE_UPPER_CH = 12;

// Active-cilium trim offsets
constexpr int8_t ACTIVE_LOWER_TRIM_DEG = -5;
constexpr int8_t ACTIVE_UPPER_TRIM_DEG = 0;

// Single neighbouring servos
constexpr uint8_t NEIGHBOUR_70_CH = 5;
constexpr uint8_t NEIGHBOUR_110_CH = 7;

// Individual neighbour trims
constexpr int8_t NEIGHBOUR_70_TRIM_DEG = 0;
constexpr int8_t NEIGHBOUR_110_TRIM_DEG = 0;

// ============================================================
// FIXED NEIGHBOUR POSITIONS
// ============================================================

constexpr uint8_t NEIGHBOUR_70_ANGLE_DEG = 90;
constexpr uint8_t NEIGHBOUR_110_ANGLE_DEG = 120;

// ============================================================
// GAIT TIMING
// ============================================================

constexpr uint32_t CYCLE_MS = 3000;
constexpr uint32_t UPDATE_MS = 5;
constexpr uint32_t STARTUP_RAMP_MS = 1500;

// ============================================================
// GLOBAL VARIABLES
// ============================================================

uint32_t gaitStartTime = 0;
uint32_t lastUpdateTime = 0;

// ============================================================
// APPLY TRIM
// ============================================================

uint8_t applyTrim(float angleDeg, int8_t trimDeg)
{
    const int16_t trimmedAngle =
        static_cast<int16_t>(round(angleDeg))
        + trimDeg;

    return static_cast<uint8_t>(
        constrain(trimmedAngle, 0, 180)
    );
}

// ============================================================
// SET THE TWO NEIGHBOURING SERVOS
// ============================================================

void setNeighboursOutOfWay()
{
    servo.setAngle(
        NEIGHBOUR_70_CH,
        applyTrim(
            NEIGHBOUR_70_ANGLE_DEG,
            NEIGHBOUR_70_TRIM_DEG
        )
    );

    servo.setAngle(
        NEIGHBOUR_110_CH,
        applyTrim(
            NEIGHBOUR_110_ANGLE_DEG,
            NEIGHBOUR_110_TRIM_DEG
        )
    );
}

// ============================================================
// RAMP ACTIVE CILIUM TO START OF GAIT
// ============================================================

void rampActiveCiliumToStart()
{
    const uint8_t targetLower =
        pgm_read_byte(&LOWER_TABLE[0]);

    const uint8_t targetUpper =
        pgm_read_byte(&UPPER_TABLE[0]);

    const float startLower = 70.0f;
    const float startUpper = 110.0f;

    const uint32_t rampStart = millis();

    while (millis() - rampStart < STARTUP_RAMP_MS)
    {
        const uint32_t elapsed =
            millis() - rampStart;

        const float fraction =
            static_cast<float>(elapsed)
            / static_cast<float>(STARTUP_RAMP_MS);

        const float smoothFraction =
            fraction
            * fraction
            * (3.0f - 2.0f * fraction);

        const float lowerAngle =
            startLower
            + (
                static_cast<float>(targetLower)
                - startLower
            ) * smoothFraction;

        const float upperAngle =
            startUpper
            + (
                static_cast<float>(targetUpper)
                - startUpper
            ) * smoothFraction;

        servo.setAngle(
            ACTIVE_LOWER_CH,
            applyTrim(
                lowerAngle,
                ACTIVE_LOWER_TRIM_DEG
            )
        );

        servo.setAngle(
            ACTIVE_UPPER_CH,
            applyTrim(
                upperAngle,
                ACTIVE_UPPER_TRIM_DEG
            )
        );

        setNeighboursOutOfWay();

        delay(5);
    }

    servo.setAngle(
        ACTIVE_LOWER_CH,
        applyTrim(
            targetLower,
            ACTIVE_LOWER_TRIM_DEG
        )
    );

    servo.setAngle(
        ACTIVE_UPPER_CH,
        applyTrim(
            targetUpper,
            ACTIVE_UPPER_TRIM_DEG
        )
    );
}

// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(115200);

    Wire.begin();

    servo.init(PCA_ADDR);
    servo.setServoPulseRange(600, 2400, 180);
    delay(500);

    setNeighboursOutOfWay();

    delay(500);

    rampActiveCiliumToStart();

    gaitStartTime = millis();
    lastUpdateTime = millis();

    Serial.println("Active cilium gait started.");
    Serial.println("Neighbour channel 10 fixed at 70 degrees.");
    Serial.println("Neighbour channel 13 fixed at 110 degrees.");
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop()
{
    const uint32_t currentTime = millis();

    if (currentTime - lastUpdateTime < UPDATE_MS)
    {
        return;
    }

    lastUpdateTime = currentTime;

    setNeighboursOutOfWay();

    const uint32_t elapsedTime =
        currentTime - gaitStartTime;

    const float phase =
        static_cast<float>(
            elapsedTime % CYCLE_MS
        )
        / static_cast<float>(CYCLE_MS);

    const float tablePosition =
        phase * static_cast<float>(GAIT_TABLE_SIZE);

    const uint16_t index0 =
        static_cast<uint16_t>(tablePosition)
        % GAIT_TABLE_SIZE;

    const uint16_t index1 =
        (index0 + 1)
        % GAIT_TABLE_SIZE;

    const float interpolationFraction =
        tablePosition
        - static_cast<float>(index0);

    const uint8_t lower0 =
        pgm_read_byte(&LOWER_TABLE[index0]);

    const uint8_t lower1 =
        pgm_read_byte(&LOWER_TABLE[index1]);

    const uint8_t upper0 =
        pgm_read_byte(&UPPER_TABLE[index0]);

    const uint8_t upper1 =
        pgm_read_byte(&UPPER_TABLE[index1]);

    const float lowerAngle =
        static_cast<float>(lower0)
        + (
            static_cast<float>(lower1)
            - static_cast<float>(lower0)
        ) * interpolationFraction;

    const float upperAngle =
        static_cast<float>(upper0)
        + (
            static_cast<float>(upper1)
            - static_cast<float>(upper0)
        ) * interpolationFraction;

    servo.setAngle(
        ACTIVE_LOWER_CH,
        applyTrim(
            lowerAngle,
            ACTIVE_LOWER_TRIM_DEG
        )
    );

    servo.setAngle(
        ACTIVE_UPPER_CH,
        applyTrim(
            upperAngle,
            ACTIVE_UPPER_TRIM_DEG
        )
    );
}