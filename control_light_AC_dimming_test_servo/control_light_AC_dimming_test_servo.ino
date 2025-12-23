// Relay pins (replacing LEDs)
const int indexRelay = 3;
const int middleRelay = 5;
const int ringRelay = 6;
const int littleRelay = 9;
const int thumbRelay = 10;

// Servo pin (kept in case you still need it)
const int servoPin = 11;

// Store relay states
bool relayStates[5] = {false, false, false, false, false};

// Last servo angle (to hold position in Normal Mode)
int lastServoAngle = 90;

#include <Servo.h>
Servo myServo;

void setup() {
  pinMode(indexRelay, OUTPUT);
  pinMode(middleRelay, OUTPUT);
  pinMode(ringRelay, OUTPUT);
  pinMode(littleRelay, OUTPUT);
  pinMode(thumbRelay, OUTPUT);

  // Initially turn all relays OFF
  digitalWrite(indexRelay, LOW);
  digitalWrite(middleRelay, LOW);
  digitalWrite(ringRelay, LOW);
  digitalWrite(littleRelay, LOW);
  digitalWrite(thumbRelay, LOW);

  myServo.attach(servoPin);
  myServo.write(lastServoAngle); // Start at middle position

  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim(); // remove extra spaces/newlines

    if (input.length() == 5 && isDigit(input[0])) {
      // --- Handle Relay control ---
      updateRelay(indexRelay, input[0], 0);
      updateRelay(middleRelay, input[1], 1);
      updateRelay(ringRelay, input[2], 2);
      updateRelay(littleRelay, input[3], 3);
      updateRelay(thumbRelay, input[4], 4);

    } else if (input.startsWith("A")) {
      // --- Handle Servo control ---
      int angle = input.substring(1).toInt(); // extract number after "A"
      angle = constrain(angle, 0, 180);       // ensure within valid range
      myServo.write(angle);
      lastServoAngle = angle;                 // save last angle for Normal Mode
    }
  }
}

// Update relay only if needed (prevents unnecessary switching)
void updateRelay(int pin, char stateChar, int index) {
  bool newState = (stateChar == '1');   // HIGH for ON, LOW for OFF
  if (newState != relayStates[index]) {
    digitalWrite(pin, newState ? HIGH : LOW);
    relayStates[index] = newState;
  }
}
