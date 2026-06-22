#include <SPI.h>
#include <Ethernet.h>
#include <ArduinoModbus.h>
#include <P1AM.h>

#include "P1AMHardware.h"
#include "SignalBroker.h"
#include "PIDController.h"
#include "SafetyInterlock.h"
#include "StorageManager.h"

// Ethernet Configuration (P1AM-ETH shield)
byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED };
IPAddress ip(192, 168, 1, 100);

EthernetServer ethServer(502); // Modbus TCP port
ModbusTCPServer modbusServer;

// Core DCS Components
P1AMHardware hw;
SignalBroker broker;
PIDController pids[4];
SafetyInterlock interlock;
StorageManager storage;

// Timing Control (10Hz / 100ms scan cycle)
unsigned long lastScanTime = 0;
const unsigned long kScanIntervalMs = 100;

// Modbus coil 2 = heater relay command from the temperature controller.
// (Coil 0 = save-to-flash, coil 1 = E-stop reset.)
const int kHeaterRelayCoil = 2;

// Helper to Pack Float into 2 Modbus registers (IEEE-754)
void WriteFloatToModbus(int regAddress, float val) {
  uint32_t raw;
  memcpy(&raw, &val, sizeof(float));
  uint16_t low = raw & 0xFFFF;
  uint16_t high = (raw >> 16) & 0xFFFF;
  modbusServer.holdingRegisterWrite(regAddress, low);
  modbusServer.holdingRegisterWrite(regAddress + 1, high);
}

// Helper to Unpack Float from 2 Modbus registers (IEEE-754)
float ReadFloatFromModbus(int regAddress) {
  uint16_t low = modbusServer.holdingRegisterRead(regAddress);
  uint16_t high = modbusServer.holdingRegisterRead(regAddress + 1);
  uint32_t raw = (static_cast<uint32_t>(high) << 16) | low;
  float val;
  memcpy(&val, &raw, sizeof(float));
  return val;
}

// Sync Modbus Configuration Registers to local configurations
void SyncModbusToDCS() {
  // 1. Sync Hardware Input routing (Registers 100-105)
  for (int i = 0; i < SignalBroker::kNumInputs; ++i) {
    int currentRegVal = modbusServer.holdingRegisterRead(100 + i);
    if (currentRegVal != broker.GetInputRouting(i)) {
      broker.SetInputRouting(i, currentRegVal);
    }
  }

  // 2. Sync Hardware Output routing (Registers 110-111)
  for (int i = 0; i < SignalBroker::kNumOutputs; ++i) {
    int currentRegVal = modbusServer.holdingRegisterRead(110 + i);
    if (currentRegVal != broker.GetOutputRouting(i)) {
      broker.SetOutputRouting(i, currentRegVal);
    }
  }

  // 3. Sync PID configurations (Registers 200-239)
  for (int i = 0; i < 4; ++i) {
    int baseReg = 200 + i * 10;
    int pv = modbusServer.holdingRegisterRead(baseReg);
    int cv = modbusServer.holdingRegisterRead(baseReg + 1);
    float sp = ReadFloatFromModbus(baseReg + 2);
    float kp = ReadFloatFromModbus(baseReg + 4);
    float ki = ReadFloatFromModbus(baseReg + 6);
    float kd = ReadFloatFromModbus(baseReg + 8);

    if (pv != pids[i].GetPvTagId()) pids[i].SetPvTagId(pv);
    if (cv != pids[i].GetCvTagId()) pids[i].SetCvTagId(cv);
    if (sp != pids[i].GetSetpoint()) pids[i].SetSetpoint(sp);
    if (kp != pids[i].GetKp()) pids[i].SetKp(kp);
    if (ki != pids[i].GetKi()) pids[i].SetKi(ki);
    if (kd != pids[i].GetKd()) pids[i].SetKd(kd);
  }

  // 4. Sync Safety Interlock limits (Registers 300-555, 8 regs / tag).
  // Layout per tag: [lolo, low, high, hihi], each as 2-register IEEE-754 float.
  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    int baseReg = 300 + i * 8;
    float loloLim = ReadFloatFromModbus(baseReg);
    float lowLim = ReadFloatFromModbus(baseReg + 2);
    float highLim = ReadFloatFromModbus(baseReg + 4);
    float hihiLim = ReadFloatFromModbus(baseReg + 6);

    if (loloLim != interlock.GetLoloLimit(i)) interlock.SetLoloLimit(i, loloLim);
    if (lowLim != interlock.GetLowLimit(i)) interlock.SetLowLimit(i, lowLim);
    if (highLim != interlock.GetHighLimit(i)) interlock.SetHighLimit(i, highLim);
    if (hihiLim != interlock.GetHihiLimit(i)) interlock.SetHihiLimit(i, hihiLim);
  }
}

// Write local configurations to Modbus registers (used on boot / load)
void SyncDCSToModbus() {
  for (int i = 0; i < SignalBroker::kNumInputs; ++i) {
    modbusServer.holdingRegisterWrite(100 + i, broker.GetInputRouting(i));
  }
  for (int i = 0; i < SignalBroker::kNumOutputs; ++i) {
    modbusServer.holdingRegisterWrite(110 + i, broker.GetOutputRouting(i));
  }
  for (int i = 0; i < 4; ++i) {
    int baseReg = 200 + i * 10;
    modbusServer.holdingRegisterWrite(baseReg, pids[i].GetPvTagId());
    modbusServer.holdingRegisterWrite(baseReg + 1, pids[i].GetCvTagId());
    WriteFloatToModbus(baseReg + 2, pids[i].GetSetpoint());
    WriteFloatToModbus(baseReg + 4, pids[i].GetKp());
    WriteFloatToModbus(baseReg + 6, pids[i].GetKi());
    WriteFloatToModbus(baseReg + 8, pids[i].GetKd());
  }
  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    int baseReg = 300 + i * 8;
    WriteFloatToModbus(baseReg, interlock.GetLoloLimit(i));
    WriteFloatToModbus(baseReg + 2, interlock.GetLowLimit(i));
    WriteFloatToModbus(baseReg + 4, interlock.GetHighLimit(i));
    WriteFloatToModbus(baseReg + 6, interlock.GetHihiLimit(i));
  }
}

void setup() {
  Serial.begin(115200);
  // Wait up to 5s for serial monitor to attach so we don't miss boot messages.
  // Note: USB CDC `if (Serial)` only blocks while a host is opening the port,
  // so this still proceeds promptly when no monitor is connected.
  unsigned long serialWaitStart = millis();
  while (!Serial && (millis() - serialWaitStart) < 5000) {
    delay(10);
  }
  Serial.println();
  Serial.println(F("=== P1AM SCADA firmware boot ==="));

  // P1AM backplane init FIRST. The P1AM library shares the SAMD21 SPI bus
  // with the W5500 Ethernet chip but doesn't use SPI.beginTransaction
  // internally; the proven workaround (per facts-engineering/P1AM#31) is to
  // initialize P1AM before Ethernet.
  Serial.println(F("[hw] P1AMHardware::Begin (backplane init)..."));
  hw.Begin();
  Serial.println(F("[hw] hardware init complete"));
  // Diagnostic: dump the signed-on module list so we can confirm the
  // P1-04THM and P1-4ADL2DAL-1 are present on the backplane. If the AO
  // module is missing, P1.writeAnalog silently returns and outputs hold
  // at their DAC power-on default of 4 mA.
  Serial.println(F("[hw] signed-on modules:"));
  P1.printModules();

  // Initialize Ethernet. The P1AM-ETH shield wires W5500 CS to pin 5.
  Ethernet.init(5);
  Serial.println(F("[eth] calling Ethernet.begin()..."));
  Ethernet.begin(mac, ip);
  Serial.print(F("[eth] hardwareStatus="));
  Serial.println(Ethernet.hardwareStatus());
  Serial.print(F("[eth] linkStatus="));
  Serial.println(Ethernet.linkStatus());
  Serial.print(F("[eth] localIP="));
  Serial.println(Ethernet.localIP());

  // Start Modbus TCP server. Done early so port 502 is reachable from SCADA
  // even if later init takes a long time.
  Serial.println(F("[mb] starting Modbus TCP server on port 502..."));
  ethServer.begin();
  if (!modbusServer.begin()) {
    Serial.println(F("[mb] FATAL: modbusServer.begin() failed -- halting"));
    while (1) {
      delay(1000); // Halt if Modbus fails to start
    }
  }
  modbusServer.configureCoils(0, 10);
  // Holding-register window: tag values (0..63), input routing (100..105),
  // output routing (110..111), PID config (200..239), 4-limit interlocks
  // (300..555 = 32 tags x 8 regs). Bump end to 560 with margin.
  modbusServer.configureHoldingRegisters(0, 560);
  Serial.println(F("[mb] Modbus TCP server started"));

  // Load saved NVRAM configuration; fall back to defaults on first boot.
  Serial.println(F("[storage] loading saved config from flash..."));
  float temp_lolo[SignalBroker::kNumTags];
  float temp_low[SignalBroker::kNumTags];
  float temp_high[SignalBroker::kNumTags];
  float temp_hihi[SignalBroker::kNumTags];
  bool loaded = storage.Load(broker, pids, temp_lolo, temp_low, temp_high, temp_hihi);
  if (loaded) {
    Serial.println(F("[storage] valid config loaded"));
    for (int i = 0; i < SignalBroker::kNumTags; ++i) {
      interlock.SetLoloLimit(i, temp_lolo[i]);
      interlock.SetLowLimit(i, temp_low[i]);
      interlock.SetHighLimit(i, temp_high[i]);
      interlock.SetHihiLimit(i, temp_hihi[i]);
    }
  } else {
    Serial.println(F("[storage] no valid config -- using power-on defaults"));
    broker.Reset();
    interlock.Reset();
    for (int i = 0; i < 4; ++i) {
      pids[i].Reset();
    }
    // Sensible power-on routing so a freshly-flashed unit (NVRAM erased by the
    // upload) boots into the bench hardware map instead of all-unmapped. An
    // all-unmapped map strands every TC/AI AND blocks recovery: the host's
    // config encoder rejects the 255 "unmapped" sentinel, so it cannot write a
    // good config back. Bench map: TC0-3 -> TAG_0..3, AI0/AI1 -> TAG_12/13,
    // AO0/AO1 <- TAG_10/11.
    broker.SetInputRouting(0, 0);
    broker.SetInputRouting(1, 1);
    broker.SetInputRouting(2, 2);
    broker.SetInputRouting(3, 3);
    broker.SetInputRouting(4, 12);
    broker.SetInputRouting(5, 13);
    broker.SetOutputRouting(0, 10);
    broker.SetOutputRouting(1, 11);
    // PID0 = power-supply current-command pass-through (CV -> AO TAG_10, unity
    // gain, PV an unrouted tag that stays 0). The host's connect-time auto-repair
    // then sees PID0 already correct and never rewrites config (which would choke
    // on the still-unmapped PID1-3). Setpoint 0 => AO idle until the PS commands.
    pids[0].SetPvTagId(30);
    pids[0].SetCvTagId(10);
    pids[0].SetKp(1.0f);
    pids[0].SetKi(0.0f);
    pids[0].SetKd(0.0f);
    pids[0].SetSetpoint(0.0f);
  }

  // Publish current config to Modbus registers.
  SyncDCSToModbus();
  Serial.println(F("[setup] complete -- entering control loop"));
}

void loop() {
  // Listen for new Modbus TCP client connections (non-blocking).
  EthernetClient newClient = ethServer.available();
  if (newClient) {
    modbusServer.accept(newClient);
  }
  modbusServer.poll();

  // Save-to-flash trigger
  if (modbusServer.coilRead(0) == 1) {
    float temp_lolo[SignalBroker::kNumTags];
    float temp_low[SignalBroker::kNumTags];
    float temp_high[SignalBroker::kNumTags];
    float temp_hihi[SignalBroker::kNumTags];
    for (int i = 0; i < SignalBroker::kNumTags; ++i) {
      temp_lolo[i] = interlock.GetLoloLimit(i);
      temp_low[i] = interlock.GetLowLimit(i);
      temp_high[i] = interlock.GetHighLimit(i);
      temp_hihi[i] = interlock.GetHihiLimit(i);
    }
    storage.Save(broker, pids, temp_lolo, temp_low, temp_high, temp_hihi);
    modbusServer.coilWrite(0, 0);
  }

  // 10 Hz control + config-sync cycle. SyncModbusToDCS is gated to this
  // timer so the SAMD21 USB CDC and Modbus library aren't starved.
  unsigned long now = millis();
  if (now - lastScanTime >= kScanIntervalMs) {
    lastScanTime = now;

    SyncModbusToDCS();
    hw.Update();
    broker.ReadHardwareInputs(hw);
    for (int i = 0; i < 4; ++i) {
      pids[i].Compute(broker, 0.1f);
    }
    interlock.Evaluate(broker, hw);
    // Heater relay (Modbus coil 2): the temperature controller commands it, but
    // the safety interlock always wins — a trip forces the relay off regardless.
    bool relay_cmd = (modbusServer.coilRead(kHeaterRelayCoil) == 1);
    hw.WriteHeaterRelay(relay_cmd && !interlock.IsTripped());

    // --- Signal diagnostics: raw 0-5 V of every analog channel (TAG_20..25) ---
    // Unscaled, for troubleshooting the analog card independent of calibration.
    // AI0..3 show the actual 0-5 V at the input terminal; AO0..1 show the
    // commanded output as 0-5 V (0 % -> 0 V, 100 % -> 5 V). These tags are
    // diagnostics only — nothing routes through them.
    for (int ch = 0; ch < 4; ++ch) {
      broker.SetTag(20 + ch, hw.ReadAnalogInputRawVolts(ch));
    }
    for (int ch = 0; ch < 2; ++ch) {
      int srcTag = broker.GetOutputRouting(ch);
      float cmdPct = (srcTag >= 0 && srcTag < SignalBroker::kNumTags)
                         ? broker.GetTag(srcTag)
                         : 0.0f;
      broker.SetTag(24 + ch, cmdPct * (5.0f / 100.0f));
    }

    for (int i = 0; i < SignalBroker::kNumTags; ++i) {
      WriteFloatToModbus(i * 2, broker.GetTag(i));
    }
  }
}
