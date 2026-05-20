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

  // 4. Sync Safety Interlock limits (Registers 300-427)
  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    int baseReg = 300 + i * 4;
    float highLim = ReadFloatFromModbus(baseReg);
    float lowLim = ReadFloatFromModbus(baseReg + 2);

    if (highLim != interlock.GetHighLimit(i)) interlock.SetHighLimit(i, highLim);
    if (lowLim != interlock.GetLowLimit(i)) interlock.SetLowLimit(i, lowLim);
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
    int baseReg = 300 + i * 4;
    WriteFloatToModbus(baseReg, interlock.GetHighLimit(i));
    WriteFloatToModbus(baseReg + 2, interlock.GetLowLimit(i));
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

  // Initialize Ethernet. The P1AM-ETH shield wires the W5500 chip-select to
  // pin 5 (this matches the library default on SAMD, but we set it explicitly
  // so the dependency is visible and survives a library default change).
  Ethernet.init(5);
  Serial.println(F("[eth] calling Ethernet.begin()..."));
  Ethernet.begin(mac, ip);
  Serial.print(F("[eth] hardwareStatus="));
  Serial.println(Ethernet.hardwareStatus());
  Serial.print(F("[eth] linkStatus="));
  Serial.println(Ethernet.linkStatus());
  Serial.print(F("[eth] localIP="));
  Serial.println(Ethernet.localIP());
  
  // Wait for Link connection
  delay(1000);

  // Start Modbus TCP server FIRST -- before any I/O / flash init that might
  // hang. Ensures port 502 is reachable from SCADA even if the backplane scan
  // or flash load below takes a long time or wedges entirely.
  Serial.println(F("[mb] starting Modbus TCP server on port 502..."));
  ethServer.begin();
  if (!modbusServer.begin()) {
    Serial.println(F("[mb] FATAL: modbusServer.begin() failed -- halting"));
    while (1) {
      delay(1000); // Halt if Modbus fails to start
    }
  }
  modbusServer.configureCoils(0, 10);
  modbusServer.configureHoldingRegisters(0, 500);
  Serial.println(F("[mb] Modbus TCP server started"));

  // Initialize hardware wrapper (backplane scan + Inhibit pin setup)
  Serial.println(F("[hw] calling P1AMHardware::Begin()..."));
  hw.Begin();
  Serial.println(F("[hw] hardware init complete"));

  // Load saved NVRAM configuration
  Serial.println(F("[storage] loading saved config from flash..."));
  float temp_high[SignalBroker::kNumTags];
  float temp_low[SignalBroker::kNumTags];

  bool loaded = storage.Load(broker, pids, temp_high, temp_low);
  if (loaded) {
    Serial.println(F("[storage] valid config loaded"));
    for (int i = 0; i < SignalBroker::kNumTags; ++i) {
      interlock.SetHighLimit(i, temp_high[i]);
      interlock.SetLowLimit(i, temp_low[i]);
    }
  } else {
    Serial.println(F("[storage] no valid config -- using defaults"));
    broker.Reset();
    interlock.Reset();
    for (int i = 0; i < 4; ++i) {
      pids[i].Reset();
    }
  }

  // Write loaded configuration values to Modbus registers
  SyncDCSToModbus();
  Serial.println(F("[setup] complete -- entering control loop"));
}

void loop() {
  // Listen for new Modbus TCP client connections (non-blocking).
  // ArduinoModbus's ModbusTCPServer::accept(Client&) takes ownership of the
  // session; poll() services it on every loop pass without blocking the 10 Hz
  // scan cycle below. This is the official ArduinoModbus TCP idiom but with
  // the while(client.connected()) blocking loop omitted so control keeps running.
  EthernetClient newClient = ethServer.available();
  if (newClient) {
    modbusServer.accept(newClient);
  }
  modbusServer.poll();

  // Trigger Save to Flash coil check
  if (modbusServer.coilRead(0) == 1) {
    float temp_high[SignalBroker::kNumTags];
    float temp_low[SignalBroker::kNumTags];
    for (int i = 0; i < SignalBroker::kNumTags; ++i) {
      temp_high[i] = interlock.GetHighLimit(i);
      temp_low[i] = interlock.GetLowLimit(i);
    }
    storage.Save(broker, pids, temp_high, temp_low);
    modbusServer.coilWrite(0, 0); // Clear trigger
  }

  // 10Hz control + config-sync logic. SyncModbusToDCS() reads ~180 holding
  // registers and unpacks IEEE-754 floats; calling it every loop iteration
  // (which on SAMD21 can be 50k+ Hz) starves USB CDC servicing and creates
  // contention with modbusServer.poll() on the same internal state. Gating to
  // 10 Hz brings load to a level the SAMD21 USB stack and Modbus library
  // can both keep up with, while still being more than fast enough to honor
  // SCADA-pushed configuration changes between control scans.
  unsigned long now = millis();
  if (now - lastScanTime >= kScanIntervalMs) {
    lastScanTime = now;

    // Pull any config changes pushed from SCADA
    SyncModbusToDCS();

    // Scan cycle
    hw.Update();
    broker.ReadHardwareInputs(hw);

    // Compute PID controllers
    for (int i = 0; i < 4; ++i) {
      pids[i].Compute(broker, 0.1f); // dt = 100ms = 0.1s
    }

    // Safety interlocks check (forces outputs to 0 + drives physical outputs)
    interlock.Evaluate(broker, hw);

    // Publish tag values to Modbus holding registers (0 to 63)
    for (int i = 0; i < SignalBroker::kNumTags; ++i) {
      WriteFloatToModbus(i * 2, broker.GetTag(i));
    }
  }
}
