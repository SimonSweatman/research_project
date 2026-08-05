#include <Arduino.h>
#include <Wire.h>

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "PCA9685.h"
#include "gait_table.h"

// ============================================================
// HARDWARE CONFIGURATION
// ============================================================

ServoDriver board1;
ServoDriver board2;

// Physical installation: cilia 1-6 are on 0x7E and cilia 7-12 are on 0x7F.
constexpr uint8_t BOARD1_ADDR = 0x7E;
constexpr uint8_t BOARD2_ADDR = 0x7F;
constexpr uint8_t CILIA_COUNT = 12;

struct CiliumChannels
{
    ServoDriver* board;
    uint8_t lowerChannel;
    uint8_t upperChannel;
};

// This is the exact channel numbering from the working set_90 sketch.
// Board 1 drives cilia 1-6; board 2 drives cilia 7-12.
CiliumChannels cilia[CILIA_COUNT] = {
    {&board1, 1,  2},   // Cilium 1
    {&board1, 3,  4},   // Cilium 2
    {&board1, 5,  6},   // Cilium 3
    {&board1, 9,  10},  // Cilium 4
    {&board1, 11, 12},  // Cilium 5
    {&board1, 13, 14},  // Cilium 6
    {&board2, 1,  2},   // Cilium 7
    {&board2, 3,  4},   // Cilium 8
    {&board2, 5,  6},   // Cilium 9
    {&board2, 9,  10},  // Cilium 10
    {&board2, 11, 12},  // Cilium 11
    {&board2, 13, 14}   // Cilium 12
};

// Experimentally fitted common conversion used by the path designer:
// PWM = 304.47 + 2.35018 * (desired angle - 90).
// Gait-table values are already converted to raw PWM by the designer, so the
// factor must not be applied again here. It is recorded here to derive the
// common physical 90-degree home position and document the active calibration.
constexpr float CALIBRATED_PWM_AT_90 = 304.47f;
constexpr float CALIBRATED_COUNTS_PER_DEG = 2.35018f;

// Retain the conservative raw-PWM bounds verified with pwm_test.ino.
constexpr uint16_t MIN_SERVO_PWM = 150;
constexpr uint16_t MAX_SERVO_PWM = 500;
constexpr uint16_t HOME_PWM =
    static_cast<uint16_t>(CALIBRATED_PWM_AT_90 + 0.5f);

// ============================================================
// MOTION SETTINGS
// ============================================================

// The PCA9685 and servos operate at 50 Hz, so one command update per 20 ms
// frame avoids needlessly saturating the I2C bus with 24 channel writes.
constexpr uint32_t SERVO_UPDATE_MS = 20;
constexpr uint32_t TRANSITION_MS = 1500;
constexpr float MIN_CYCLE_SECONDS = 0.5f;
constexpr float MAX_CYCLE_SECONDS = 120.0f;

uint32_t cycleDurationMs = 5000;
float adjacentPhaseShiftDeg = 0.0f;
float globalPhase = 0.0f;

static_assert(CILIA_COUNT == 12, "The channel map must contain 12 cilia.");
static_assert(GAIT_TABLE_SIZE >= 2, "The gait table needs at least two entries.");
static_assert(
    sizeof(LOWER_TABLE) / sizeof(LOWER_TABLE[0]) == GAIT_TABLE_SIZE,
    "LOWER_TABLE length does not match GAIT_TABLE_SIZE."
);
static_assert(
    sizeof(UPPER_TABLE) / sizeof(UPPER_TABLE[0]) == GAIT_TABLE_SIZE,
    "UPPER_TABLE length does not match GAIT_TABLE_SIZE."
);
static_assert(
    GAIT_PWM_FREQUENCY_HZ == 50,
    "The gait table must be exported for 50 Hz PWM."
);

// ============================================================
// RUN STATE
// ============================================================

enum class RunState : uint8_t
{
    STOPPED,
    STARTING,
    RUNNING,
    PAUSED,
    STOPPING
};

RunState runState = RunState::STOPPED;
bool pausedFromRunning = false;

uint32_t lastMotionUpdateMs = 0;
uint32_t transitionStartMs = 0;

uint16_t currentLowerPwm[CILIA_COUNT];
uint16_t currentUpperPwm[CILIA_COUNT];
uint16_t transitionStartLower[CILIA_COUNT];
uint16_t transitionStartUpper[CILIA_COUNT];
uint16_t transitionTargetLower[CILIA_COUNT];
uint16_t transitionTargetUpper[CILIA_COUNT];

// ============================================================
// SERIAL INPUT
// ============================================================

constexpr uint8_t SERIAL_BUFFER_SIZE = 48;
char serialBuffer[SERIAL_BUFFER_SIZE];
uint8_t serialLength = 0;
bool serialOverflow = false;

// ============================================================
// PWM AND GAIT HELPERS
// ============================================================

uint16_t clampPwm(float pwmCount)
{
    const int32_t rounded = static_cast<int32_t>(pwmCount + 0.5f);
    return static_cast<uint16_t>(constrain(
        rounded,
        static_cast<int32_t>(MIN_SERVO_PWM),
        static_cast<int32_t>(MAX_SERVO_PWM)
    ));
}

uint16_t readGaitPwm(const uint16_t* table, uint16_t index)
{
    return pgm_read_word(&table[index]);
}

float wrapPhase(float phase)
{
    phase -= floorf(phase);
    if (phase < 0.0f)
    {
        phase += 1.0f;
    }
    return phase;
}

void sampleGait(float phase, float& lowerPwm, float& upperPwm)
{
    const float wrappedPhase = wrapPhase(phase);
    const float tablePosition =
        wrappedPhase * static_cast<float>(GAIT_TABLE_SIZE);
    const uint16_t index0 =
        static_cast<uint16_t>(tablePosition) % GAIT_TABLE_SIZE;
    const uint16_t index1 = (index0 + 1) % GAIT_TABLE_SIZE;
    const float fraction = tablePosition - static_cast<float>(index0);

    const uint16_t lower0 = readGaitPwm(LOWER_TABLE, index0);
    const uint16_t lower1 = readGaitPwm(LOWER_TABLE, index1);
    const uint16_t upper0 = readGaitPwm(UPPER_TABLE, index0);
    const uint16_t upper1 = readGaitPwm(UPPER_TABLE, index1);

    lowerPwm = static_cast<float>(lower0)
        + (static_cast<float>(lower1) - static_cast<float>(lower0)) * fraction;
    upperPwm = static_cast<float>(upper0)
        + (static_cast<float>(upper1) - static_cast<float>(upper0)) * fraction;
}

void writeCilium(uint8_t index, float lowerPwm, float upperPwm)
{
    const uint16_t safeLower = clampPwm(lowerPwm);
    const uint16_t safeUpper = clampPwm(upperPwm);

    cilia[index].board->setPwm(cilia[index].lowerChannel, 0, safeLower);
    cilia[index].board->setPwm(cilia[index].upperChannel, 0, safeUpper);

    currentLowerPwm[index] = safeLower;
    currentUpperPwm[index] = safeUpper;
}

void commandAllHome()
{
    for (uint8_t index = 0; index < CILIA_COUNT; ++index)
    {
        writeCilium(index, HOME_PWM, HOME_PWM);
    }
}

void calculateGaitTargets(
    float basePhase,
    uint16_t* lowerTargets,
    uint16_t* upperTargets
)
{
    for (uint8_t index = 0; index < CILIA_COUNT; ++index)
    {
        const float ciliumPhase = basePhase
            + static_cast<float>(index) * adjacentPhaseShiftDeg / 360.0f;
        float lowerPwm = 0.0f;
        float upperPwm = 0.0f;
        sampleGait(ciliumPhase, lowerPwm, upperPwm);
        lowerTargets[index] = clampPwm(lowerPwm);
        upperTargets[index] = clampPwm(upperPwm);
    }
}

void commandArrayAtPhase(float basePhase)
{
    for (uint8_t index = 0; index < CILIA_COUNT; ++index)
    {
        const float ciliumPhase = basePhase
            + static_cast<float>(index) * adjacentPhaseShiftDeg / 360.0f;
        float lowerPwm = 0.0f;
        float upperPwm = 0.0f;
        sampleGait(ciliumPhase, lowerPwm, upperPwm);
        writeCilium(index, lowerPwm, upperPwm);
    }
}

// ============================================================
// NON-BLOCKING START/STOP TRANSITIONS
// ============================================================

void copyCurrentToTransitionStart()
{
    for (uint8_t index = 0; index < CILIA_COUNT; ++index)
    {
        transitionStartLower[index] = currentLowerPwm[index];
        transitionStartUpper[index] = currentUpperPwm[index];
    }
}

void beginStartTransition()
{
    globalPhase = 0.0f;
    copyCurrentToTransitionStart();
    calculateGaitTargets(
        globalPhase,
        transitionTargetLower,
        transitionTargetUpper
    );
    transitionStartMs = millis();
    runState = RunState::STARTING;
    pausedFromRunning = false;
    Serial.println(F("Starting: moving smoothly from the current pose to the gait."));
}

void beginStopTransition()
{
    copyCurrentToTransitionStart();
    for (uint8_t index = 0; index < CILIA_COUNT; ++index)
    {
        transitionTargetLower[index] = HOME_PWM;
        transitionTargetUpper[index] = HOME_PWM;
    }
    transitionStartMs = millis();
    runState = RunState::STOPPING;
    pausedFromRunning = false;
    Serial.println(F("Stopping: returning all cilia smoothly to 90 degrees."));
}

void updateTransition(uint32_t now)
{
    const uint32_t elapsed = now - transitionStartMs;
    const float fraction = min(
        1.0f,
        static_cast<float>(elapsed) / static_cast<float>(TRANSITION_MS)
    );
    const float smoothFraction =
        fraction * fraction * (3.0f - 2.0f * fraction);

    for (uint8_t index = 0; index < CILIA_COUNT; ++index)
    {
        const float lowerPwm = static_cast<float>(transitionStartLower[index])
            + (static_cast<float>(transitionTargetLower[index])
               - static_cast<float>(transitionStartLower[index])) * smoothFraction;
        const float upperPwm = static_cast<float>(transitionStartUpper[index])
            + (static_cast<float>(transitionTargetUpper[index])
               - static_cast<float>(transitionStartUpper[index])) * smoothFraction;
        writeCilium(index, lowerPwm, upperPwm);
    }

    if (fraction < 1.0f)
    {
        return;
    }

    if (runState == RunState::STARTING)
    {
        runState = RunState::RUNNING;
        lastMotionUpdateMs = now;
        Serial.println(F("RUNNING"));
    }
    else
    {
        runState = RunState::STOPPED;
        globalPhase = 0.0f;
        Serial.println(F("STOPPED: all cilia are holding at 90 degrees."));
    }
}

void updateRunningMotion(uint32_t now)
{
    const uint32_t elapsed = now - lastMotionUpdateMs;
    lastMotionUpdateMs = now;
    globalPhase = wrapPhase(
        globalPhase
        + static_cast<float>(elapsed) / static_cast<float>(cycleDurationMs)
    );
    commandArrayAtPhase(globalPhase);
}

// ============================================================
// SERIAL COMMANDS
// ============================================================

void printHelp()
{
    Serial.println(F("Commands:"));
    Serial.println(F("  start          - start a new gait, or resume a pause"));
    Serial.println(F("  pause          - hold the current positions"));
    Serial.println(F("  stop           - return smoothly to 90 degrees and hold"));
    Serial.println(F("  home           - same action as stop"));
    Serial.println(F("  phase <deg>    - adjacent phase shift, 0 to 360 (while stopped)"));
    Serial.println(F("  speed <sec>    - cycle duration in seconds, 0.5 to 120"));
    Serial.println(F("  status         - show state and settings"));
    Serial.println(F("  help           - show this list"));
}

void printState()
{
    Serial.print(F("State: "));
    switch (runState)
    {
        case RunState::STOPPED:  Serial.println(F("STOPPED"));  break;
        case RunState::STARTING: Serial.println(F("STARTING")); break;
        case RunState::RUNNING:  Serial.println(F("RUNNING"));  break;
        case RunState::PAUSED:   Serial.println(F("PAUSED"));   break;
        case RunState::STOPPING: Serial.println(F("STOPPING")); break;
    }
    Serial.print(F("Cycle duration: "));
    Serial.print(static_cast<float>(cycleDurationMs) / 1000.0f, 3);
    Serial.println(F(" s"));
    Serial.print(F("Adjacent phase shift: "));
    Serial.print(adjacentPhaseShiftDeg, 2);
    Serial.println(F(" deg"));
    Serial.print(F("Current global phase: "));
    Serial.print(globalPhase * 360.0f, 1);
    Serial.println(F(" deg"));
}

bool parseNumber(const char* text, float& value)
{
    if (text == nullptr || *text == '\0')
    {
        return false;
    }

    char* end = nullptr;
    // AVR-libc exposes strtod rather than strtof; on the Uno, double and
    // float are both 32-bit values.
    value = static_cast<float>(strtod(text, &end));
    if (end == text)
    {
        return false;
    }
    while (*end != '\0' && isspace(static_cast<unsigned char>(*end)))
    {
        ++end;
    }
    return *end == '\0';
}

void processCommand(char* line)
{
    while (*line != '\0' && isspace(static_cast<unsigned char>(*line)))
    {
        ++line;
    }
    size_t length = strlen(line);
    while (length > 0 && isspace(static_cast<unsigned char>(line[length - 1])))
    {
        line[--length] = '\0';
    }
    for (size_t index = 0; index < length; ++index)
    {
        line[index] = static_cast<char>(
            tolower(static_cast<unsigned char>(line[index]))
        );
    }

    char* argument = strchr(line, ' ');
    if (argument != nullptr)
    {
        *argument++ = '\0';
        while (*argument != '\0' && isspace(static_cast<unsigned char>(*argument)))
        {
            ++argument;
        }
    }

    if (strcmp(line, "start") == 0)
    {
        if (runState == RunState::RUNNING || runState == RunState::STARTING)
        {
            Serial.println(F("The gait is already starting or running."));
        }
        else if (runState == RunState::PAUSED && pausedFromRunning)
        {
            runState = RunState::RUNNING;
            lastMotionUpdateMs = millis();
            Serial.println(F("RUNNING: resumed from the held position."));
        }
        else
        {
            beginStartTransition();
        }
        return;
    }

    if (strcmp(line, "pause") == 0)
    {
        if (runState == RunState::RUNNING)
        {
            runState = RunState::PAUSED;
            pausedFromRunning = true;
            Serial.println(F("PAUSED: current positions are being held."));
        }
        else if (runState == RunState::STARTING || runState == RunState::STOPPING)
        {
            runState = RunState::PAUSED;
            pausedFromRunning = false;
            Serial.println(F("PAUSED during transition: current positions are held."));
        }
        else
        {
            Serial.println(F("Nothing is currently moving."));
        }
        return;
    }

    if (strcmp(line, "stop") == 0 || strcmp(line, "home") == 0)
    {
        if (runState == RunState::STOPPED)
        {
            Serial.println(F("Already stopped and holding at 90 degrees."));
        }
        else
        {
            beginStopTransition();
        }
        return;
    }

    if (strcmp(line, "phase") == 0)
    {
        float value = 0.0f;
        if (!parseNumber(argument, value) || value < 0.0f || value > 360.0f)
        {
            Serial.println(F("Use: phase <0 to 360 degrees>"));
        }
        else if (runState != RunState::STOPPED)
        {
            Serial.println(F("Stop the array before changing phase to avoid servo jumps."));
        }
        else
        {
            adjacentPhaseShiftDeg = value;
            Serial.print(F("Adjacent phase shift set to "));
            Serial.print(adjacentPhaseShiftDeg, 2);
            Serial.println(F(" degrees."));
        }
        return;
    }

    if (strcmp(line, "speed") == 0 || strcmp(line, "period") == 0)
    {
        float seconds = 0.0f;
        if (!parseNumber(argument, seconds)
            || seconds < MIN_CYCLE_SECONDS
            || seconds > MAX_CYCLE_SECONDS)
        {
            Serial.println(F("Use: speed <cycle seconds from 0.5 to 120>"));
        }
        else
        {
            cycleDurationMs = static_cast<uint32_t>(seconds * 1000.0f + 0.5f);
            Serial.print(F("Cycle duration set to "));
            Serial.print(static_cast<float>(cycleDurationMs) / 1000.0f, 3);
            Serial.println(F(" seconds."));
        }
        return;
    }

    if (strcmp(line, "status") == 0)
    {
        printState();
        return;
    }

    if (strcmp(line, "help") == 0 || strcmp(line, "?") == 0)
    {
        printHelp();
        return;
    }

    if (*line != '\0')
    {
        Serial.println(F("Unknown command. Type help for the command list."));
    }
}

void readSerialCommands()
{
    while (Serial.available() > 0)
    {
        const char incoming = static_cast<char>(Serial.read());
        if (incoming == '\r' || incoming == '\n')
        {
            if (serialOverflow)
            {
                Serial.println(F("Command too long. Type help for valid commands."));
            }
            else if (serialLength > 0)
            {
                serialBuffer[serialLength] = '\0';
                processCommand(serialBuffer);
            }
            serialLength = 0;
            serialOverflow = false;
        }
        else if (!serialOverflow)
        {
            if (serialLength < SERIAL_BUFFER_SIZE - 1)
            {
                serialBuffer[serialLength++] = incoming;
            }
            else
            {
                serialOverflow = true;
            }
        }
    }
}

// ============================================================
// SETUP AND MAIN LOOP
// ============================================================

void setup()
{
    Serial.begin(115200);
    Wire.begin();

    board1.init(BOARD1_ADDR);
    board2.init(BOARD2_ADDR);
    board1.setServoPulseRange(500, 2500, 180);
    board2.setServoPulseRange(500, 2500, 180);
    board1.setFrequency(GAIT_PWM_FREQUENCY_HZ);
    board2.setFrequency(GAIT_PWM_FREQUENCY_HZ);

    delay(500);
    commandAllHome();

    Serial.println();
    Serial.println(F("12-cilia controller ready."));
    Serial.println(F("All cilia are holding at 90 degrees; the gait is NOT running."));
    Serial.println(F("Type help for commands, then set phase/speed and type start."));
    printState();
}

void loop()
{
    readSerialCommands();

    const uint32_t now = millis();
    if (now - lastMotionUpdateMs < SERVO_UPDATE_MS)
    {
        return;
    }

    if (runState == RunState::STARTING || runState == RunState::STOPPING)
    {
        lastMotionUpdateMs = now;
        updateTransition(now);
    }
    else if (runState == RunState::RUNNING)
    {
        // updateRunningMotion() measures elapsed time before updating this
        // timestamp, so the global gait phase advances correctly.
        updateRunningMotion(now);
    }
    else
    {
        lastMotionUpdateMs = now;
    }
}
