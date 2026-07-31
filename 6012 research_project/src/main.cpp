#include <Arduino.h>
#include <Wire.h>

#include "PCA9685.h"
#include "gait_table.h"

ServoDriver servo;

// ============================================================
// HARDWARE SETTINGS
// ============================================================

constexpr uint8_t PCA_ADDR = 0x7E;
constexpr uint16_t PWM_FREQUENCY_HZ = 50;

// Active cilium
constexpr uint8_t ACTIVE_LOWER_CH = 11;
constexpr uint8_t ACTIVE_UPPER_CH = 12;

// Trims are now PWM counts, not degrees
constexpr int16_t ACTIVE_LOWER_TRIM_COUNTS = 0;
constexpr int16_t ACTIVE_UPPER_TRIM_COUNTS = 0;

// Single neighbouring servos
constexpr uint8_t NEIGHBOUR_A_CH = 9;
constexpr uint8_t NEIGHBOUR_B_CH = 7;

constexpr int16_t NEIGHBOUR_A_TRIM_COUNTS = 0;
constexpr int16_t NEIGHBOUR_B_TRIM_COUNTS = 0;

// ============================================================
// PWM CONVERSION AND SAFE RANGE
//
// These values deliberately reproduce the installed Seeed library's
// effective setAngle() conversion after its default
// setServoPulseRange(500, 2500, 180) call:
//
//     0 degrees   -> 122 counts
//     90 degrees  -> 302 counts
//     180 degrees -> 482 counts
//
// The library performs integer division internally, giving 2 counts/degree.
// Raw-PWM gait tables can use individual counts between these limits.
// ============================================================

constexpr uint16_t SERVO_ZERO_DEG_PWM = 122;
constexpr uint16_t SERVO_COUNTS_PER_DEG = 2;
constexpr uint16_t MIN_SERVO_PWM = 122;
constexpr uint16_t MAX_SERVO_PWM = 482;

// Empirically selected fixed neighbour positions. Under the conversion above,
// 284 counts is approximately 81 degrees and 330 is approximately 104 degrees.
constexpr uint16_t NEIGHBOUR_A_PWM = 284;
constexpr uint16_t NEIGHBOUR_B_PWM = 330;

// Both active joints start from the exact setAngle(90) equivalent.
constexpr uint16_t START_LOWER_PWM = 302;
constexpr uint16_t START_UPPER_PWM = 302;

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

static_assert(GAIT_TABLE_SIZE >= 2, "The gait table needs at least two entries.");
static_assert(
    sizeof(LOWER_TABLE) / sizeof(LOWER_TABLE[0]) == GAIT_TABLE_SIZE,
    "LOWER_TABLE length does not match GAIT_TABLE_SIZE."
);
static_assert(
    sizeof(UPPER_TABLE) / sizeof(UPPER_TABLE[0]) == GAIT_TABLE_SIZE,
    "UPPER_TABLE length does not match GAIT_TABLE_SIZE."
);

// ============================================================
// READ EITHER SUPPORTED GAIT-TABLE FORMAT
// ============================================================

uint16_t degreeCommandToPwm(uint8_t angleDegrees)
{
    const uint16_t limitedAngle =
        min(static_cast<uint16_t>(angleDegrees), 180U);

    return SERVO_ZERO_DEG_PWM
        + limitedAngle * SERVO_COUNTS_PER_DEG;
}

// Compatibility with the earlier uint8_t degree-table exporter.
uint16_t readGaitPwm(const uint8_t* table, uint16_t index)
{
    return degreeCommandToPwm(pgm_read_byte(&table[index]));
}

// Preferred format: raw uint16_t PCA9685 OFF counts.
uint16_t readGaitPwm(const uint16_t* table, uint16_t index)
{
    return pgm_read_word(&table[index]);
}

// ============================================================
// APPLY PWM TRIM
// ============================================================

uint16_t applyPwmTrim(float pwmCount, int16_t trimCounts)
{
    const int32_t trimmedCount =
        static_cast<int32_t>(round(pwmCount))
        + trimCounts;

    return static_cast<uint16_t>(
        constrain(
            trimmedCount,
            static_cast<int32_t>(MIN_SERVO_PWM),
            static_cast<int32_t>(MAX_SERVO_PWM)
        )
    );
}

// ============================================================
// WRITE A PWM COUNT
// ============================================================

void setServoPwm(
    uint8_t channel,
    float pwmCount,
    int16_t trimCounts = 0
)
{
    servo.setPwm(
        channel,
        0,
        applyPwmTrim(pwmCount, trimCounts)
    );
}

// ============================================================
// SET THE TWO NEIGHBOURING SERVOS
// ============================================================

void setNeighboursOutOfWay()
{
    setServoPwm(
        NEIGHBOUR_A_CH,
        NEIGHBOUR_A_PWM,
        NEIGHBOUR_A_TRIM_COUNTS
    );

    setServoPwm(
        NEIGHBOUR_B_CH,
        NEIGHBOUR_B_PWM,
        NEIGHBOUR_B_TRIM_COUNTS
    );
}

// ============================================================
// RAMP ACTIVE CILIUM TO START OF GAIT
// ============================================================

void rampActiveCiliumToStart()
{
    const uint16_t targetLower =
        readGaitPwm(LOWER_TABLE, 0);

    const uint16_t targetUpper =
        readGaitPwm(UPPER_TABLE, 0);

    const uint32_t rampStart = millis();

    while (millis() - rampStart < STARTUP_RAMP_MS)
    {
        const uint32_t elapsed =
            millis() - rampStart;

        const float fraction =
            static_cast<float>(elapsed)
            / static_cast<float>(STARTUP_RAMP_MS);

        // Smoothstep ramp
        const float smoothFraction =
            fraction
            * fraction
            * (3.0f - 2.0f * fraction);

        const float lowerPwm =
            static_cast<float>(START_LOWER_PWM)
            + (
                static_cast<float>(targetLower)
                - static_cast<float>(START_LOWER_PWM)
            ) * smoothFraction;

        const float upperPwm =
            static_cast<float>(START_UPPER_PWM)
            + (
                static_cast<float>(targetUpper)
                - static_cast<float>(START_UPPER_PWM)
            ) * smoothFraction;

        setServoPwm(
            ACTIVE_LOWER_CH,
            lowerPwm,
            ACTIVE_LOWER_TRIM_COUNTS
        );

        setServoPwm(
            ACTIVE_UPPER_CH,
            upperPwm,
            ACTIVE_UPPER_TRIM_COUNTS
        );

        delay(5);
    }

    setServoPwm(
        ACTIVE_LOWER_CH,
        targetLower,
        ACTIVE_LOWER_TRIM_COUNTS
    );

    setServoPwm(
        ACTIVE_UPPER_CH,
        targetUpper,
        ACTIVE_UPPER_TRIM_COUNTS
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

    // The table PWM counts must have been calculated for 50 Hz.
    servo.setFrequency(PWM_FREQUENCY_HZ);

    delay(500);

    setNeighboursOutOfWay();

    delay(500);

    rampActiveCiliumToStart();

    gaitStartTime = millis();
    lastUpdateTime = millis();

    Serial.println("Active cilium PWM gait started.");
    Serial.println("Neighbour channel 9 fixed at 284 PWM counts.");
    Serial.println("Neighbour channel 7 fixed at 330 PWM counts.");

    if (sizeof(LOWER_TABLE[0]) == sizeof(uint16_t))
    {
        Serial.println("Gait input: uint16_t raw-PWM table (preferred).");
    }
    else
    {
        Serial.println("Gait input: legacy uint8_t degree table converted to PWM.");
    }
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

    const uint32_t elapsedTime =
        currentTime - gaitStartTime;

    const float phase =
        static_cast<float>(elapsedTime % CYCLE_MS)
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

    // Read raw PWM values, or safely convert a legacy degree table.
    const uint16_t lower0 =
        readGaitPwm(LOWER_TABLE, index0);

    const uint16_t lower1 =
        readGaitPwm(LOWER_TABLE, index1);

    const uint16_t upper0 =
        readGaitPwm(UPPER_TABLE, index0);

    const uint16_t upper1 =
        readGaitPwm(UPPER_TABLE, index1);

    // Interpolate directly between PWM counts
    const float lowerPwm =
        static_cast<float>(lower0)
        + (
            static_cast<float>(lower1)
            - static_cast<float>(lower0)
        ) * interpolationFraction;

    const float upperPwm =
        static_cast<float>(upper0)
        + (
            static_cast<float>(upper1)
            - static_cast<float>(upper0)
        ) * interpolationFraction;

    setServoPwm(
        ACTIVE_LOWER_CH,
        lowerPwm,
        ACTIVE_LOWER_TRIM_COUNTS
    );

    setServoPwm(
        ACTIVE_UPPER_CH,
        upperPwm,
        ACTIVE_UPPER_TRIM_COUNTS
    );
}
