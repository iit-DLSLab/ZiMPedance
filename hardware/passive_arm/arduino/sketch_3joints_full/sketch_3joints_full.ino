// Three Bourns absolute encoder reader for Arduino Serial Plotter
//
// Joint 1 measured mapping:
// pin 1 blue  -> DI  -> D8 set LOW
// pin 2 grey  -> CLK -> D11
// pin 3 black -> GND -> GND
// pin 4 white -> DO  -> D10
// pin 5 pink  -> VCC -> 5V
// pin 6 brown -> CS  -> D9
//
// Joint 2:
// pin 1 blue  -> DI  -> D4 set LOW
// pin 2 grey  -> CLK -> D7
// pin 3 black -> GND -> GND
// pin 4 white -> DO  -> D6
// pin 5 pink  -> VCC -> 5V
// pin 6 brown -> CS  -> D5
//
// Joint 3:
// pin 1 blue  -> DI  -> GND
// pin 2 grey  -> CLK -> A0
// pin 3 black -> GND -> GND
// pin 4 white -> DO  -> A1
// pin 5 pink  -> VCC -> 5V
// pin 6 brown -> CS  -> A2

// ---------------- Joint 1 pins ----------------
const int J1_DI  = 8;
const int J1_CLK = 11;
const int J1_DO  = 10;
const int J1_CS  = 9;

// ---------------- Joint 2 pins ----------------
const int J2_DI  = 4;
const int J2_CLK = 7;
const int J2_DO  = 6;
const int J2_CS  = 5;

// ---------------- Joint 3 pins ----------------
const int J3_CLK = A0;
const int J3_DO  = A1;
const int J3_CS  = A2;

// ---------------- Encoder config ----------------
const int COUNTS_PER_REV = 1024;

const bool CS_ACTIVE_HIGH = false;
const bool READ_AFTER_CLK_HIGH = true;
const bool USE_TOP_10_BITS = true;

// true  = Arduino Serial Plotter output
// false = ROS/Python output: rad1<TAB>rad2<TAB>rad3
const bool PLOTTER_MODE = false;

uint16_t readEncoderRaw16(int pin_clk, int pin_do, int pin_cs)
{
  uint16_t raw = 0;

  int cs_active = CS_ACTIVE_HIGH ? HIGH : LOW;
  int cs_idle   = CS_ACTIVE_HIGH ? LOW  : HIGH;

  digitalWrite(pin_cs, cs_idle);
  digitalWrite(pin_clk, LOW);
  delayMicroseconds(5);

  digitalWrite(pin_cs, cs_active);
  delayMicroseconds(5);

  for (int i = 0; i < 16; i++)
  {
    if (READ_AFTER_CLK_HIGH)
    {
      digitalWrite(pin_clk, HIGH);
      delayMicroseconds(5);

      raw <<= 1;
      if (digitalRead(pin_do))
      {
        raw |= 1;
      }

      digitalWrite(pin_clk, LOW);
      delayMicroseconds(5);
    }
    else
    {
      digitalWrite(pin_clk, HIGH);
      delayMicroseconds(5);
      digitalWrite(pin_clk, LOW);
      delayMicroseconds(5);

      raw <<= 1;
      if (digitalRead(pin_do))
      {
        raw |= 1;
      }
    }
  }

  digitalWrite(pin_cs, cs_idle);
  delayMicroseconds(5);

  return raw;
}

int extractCount(uint16_t raw)
{
  if (USE_TOP_10_BITS)
  {
    return (raw >> 6) & 0x03FF;
  }
  else
  {
    return raw & 0x03FF;
  }
}

float countToRadians(int count)
{
  return ((float)count / (float)COUNTS_PER_REV) * 2.0 * PI;
}

void setupEncoderPins(int pin_clk, int pin_do, int pin_cs)
{
  pinMode(pin_clk, OUTPUT);
  pinMode(pin_cs, OUTPUT);
  pinMode(pin_do, INPUT);

  digitalWrite(pin_clk, LOW);
  digitalWrite(pin_cs, CS_ACTIVE_HIGH ? LOW : HIGH);
}

void setup()
{
  // DI pins for single-sensor mode
  pinMode(J1_DI, OUTPUT);
  digitalWrite(J1_DI, LOW);

  pinMode(J2_DI, OUTPUT);
  digitalWrite(J2_DI, LOW);

  setupEncoderPins(J1_CLK, J1_DO, J1_CS);
  setupEncoderPins(J2_CLK, J2_DO, J2_CS);
  setupEncoderPins(J3_CLK, J3_DO, J3_CS);

  Serial.begin(115200);
  delay(1000);
}

void loop()
{
  uint16_t raw1 = readEncoderRaw16(J1_CLK, J1_DO, J1_CS);
  uint16_t raw2 = readEncoderRaw16(J2_CLK, J2_DO, J2_CS);
  uint16_t raw3 = readEncoderRaw16(J3_CLK, J3_DO, J3_CS);

  int count1 = extractCount(raw1);
  int count2 = extractCount(raw2);
  int count3 = extractCount(raw3);

  float rad1 = countToRadians(count1);
  float rad2 = countToRadians(count2);
  float rad3 = countToRadians(count3);

  if (PLOTTER_MODE)
  {
    Serial.print("joint1_count:");
    Serial.print(count1);
    Serial.print("\t");

    Serial.print("joint2_count:");
    Serial.print(count2);
    Serial.print("\t");

    Serial.print("joint3_count:");
    Serial.println(count3);
  }
  else
  {
    // ROS/Python mode
    Serial.print(rad1, 6);
    Serial.print("\t");
    Serial.print(rad2, 6);
    Serial.print("\t");
    Serial.println(rad3, 6);
  }

  delay(50);
}