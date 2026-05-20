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

  // Initialize Ethernet
  Ethernet.begin(mac, ip);
  
  // Wait for Link connection
  delay(1000);

  // Initialize hardware wrapper
  hw.Begin();

  // Load saved NVRAM configuration
  float temp_high[SignalBroker::kNumTags];
  float temp_low[SignalBroker::kNumTags];
  
  bool loaded = storage.Load(broker, pids, temp_high, temp_low);
  if (loaded) {
    for (int i = 0; i < SignalBroker::kNumTags; ++i) {
      interlock.SetHighLimit(i, temp_high[i]);
      interlock.SetLowLimit(i, temp_low[i]);
    }
  } else {
    // Apply standard default values to prevent immediate trip
    broker.Reset();
    interlock.Reset();
    for (int i = 0; i < 4; ++i) {
      pids[i].Reset();
    }
  }

  // Start Modbus TCP server
  if (!modbusServer.begin()) {
    while (1) {
      delay(1000); // Halt if Modbus fails to start
    }
  }

  // Configure Modbus registers
  modbusServer.configureCoils(0, 10);
  modbusServer.configureHoldingRegisters(0, 500);

  // Write loaded configuration values to Modbus registers
  SyncDCSToModbus();
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

  // Sync any updates sent via SCADA
  SyncModbusToDCS();

  // 10Hz control loop logic
  unsigned long now = millis();
  if (now - lastScanTime >= kScanIntervalMs) {
    lastScanTime = now;

    // Scan cycle
    hw.Update();
    broker.ReadHardwareInputs(hw);

    // Compute PID controllers
    for (int i = 0; i < 4; ++i) {
      pids[i].Compute(broker, 0.1f); // dt = 100ms = 0.1s
    }

    // Safety interlocks check (handles forcing outputs to 0 and driving physical outputs)
    interlock.Evaluate(broker, hw);

    // Publish tag values to Modbus holding registers (0 to 63)
    for (int i = 0; i < SignalBroker::kNumTags; ++i) {
      WriteFloatToModbus(i * 2, broker.GetTag(i));
    }
  }
}
