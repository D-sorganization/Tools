// Minimal Ethernet probe for P1AM-100 + P1AM-ETH shield.
// Tries to detect the W5500 chip and report link state. Does NOT call any
// blocking init that could wedge USB CDC if the chip is unreachable.

#include <SPI.h>
#include <Ethernet.h>
#include <ArduinoModbus.h>
#include <P1AM.h>
#include <FlashStorage.h>

byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED };
IPAddress ip(192, 168, 1, 100);

EthernetServer ethServer(502);
ModbusTCPServer modbusServer;

// Same shape as the main firmware's ConfigStruct, but minimal so we can isolate
// whether FlashStorage's static instantiation is what wedges things.
struct ProbeFlash {
  int magic;
  uint8_t payload[256];
};
FlashStorage(probeFlash, ProbeFlash);

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 5000) { delay(10); }

  Serial.println();
  Serial.println(F("=== P1AM-ETH probe ==="));

  Serial.println(F("[FS] reading FlashStorage..."));
  ProbeFlash f = probeFlash.read();
  Serial.print(F("[FS] magic = 0x")); Serial.println(f.magic, HEX);

  Serial.println(F("[0] calling P1.init() -- backplane module scan"));
  uint8_t modules = P1.init();
  Serial.print(F("[0] P1.init() returned ")); Serial.print(modules); Serial.println(F(" modules"));

  Serial.println(F("[1] calling Ethernet.init(5)"));
  Ethernet.init(5);
  Serial.println(F("[1] Ethernet.init returned"));

  Serial.println(F("[2] calling Ethernet.begin(mac, ip) -- static IP, no DHCP"));
  Ethernet.begin(mac, ip);
  Serial.println(F("[2] Ethernet.begin returned"));

  Serial.print(F("[3] hardwareStatus = "));
  EthernetHardwareStatus hw = Ethernet.hardwareStatus();
  switch (hw) {
    case EthernetNoHardware: Serial.println(F("NoHardware (SPI talk failed)")); break;
    case EthernetW5100:      Serial.println(F("W5100")); break;
    case EthernetW5200:      Serial.println(F("W5200")); break;
    case EthernetW5500:      Serial.println(F("W5500 OK")); break;
    default:                 Serial.println((int)hw); break;
  }

  Serial.print(F("[4] linkStatus = "));
  EthernetLinkStatus link = Ethernet.linkStatus();
  switch (link) {
    case LinkON:    Serial.println(F("ON")); break;
    case LinkOFF:   Serial.println(F("OFF")); break;
    case Unknown:   Serial.println(F("Unknown")); break;
  }

  Serial.print(F("[5] localIP = "));
  Serial.println(Ethernet.localIP());

  Serial.println(F("[6] starting EthernetServer on port 502"));
  ethServer.begin();
  Serial.println(F("[6] EthernetServer started"));

  Serial.println(F("[7] starting ModbusTCPServer"));
  if (!modbusServer.begin()) {
    Serial.println(F("[7] modbusServer.begin() FAILED"));
  } else {
    Serial.println(F("[7] ModbusTCPServer started"));
    modbusServer.configureHoldingRegisters(0, 16);
    modbusServer.holdingRegisterWrite(0, 0xBEEF);  // sentinel
    modbusServer.holdingRegisterWrite(1, 0xCAFE);
  }

  Serial.println(F("=== probe complete -- entering 1Hz heartbeat + Modbus poll ==="));
}

void loop() {
  EthernetClient client = ethServer.available();
  if (client) {
    modbusServer.accept(client);
  }
  modbusServer.poll();

  // 10Hz scan: do the same kind of P1 backplane I/O the real firmware does
  static unsigned long lastScan = 0;
  if (millis() - lastScan >= 100) {
    lastScan = millis();
    // Read thermocouples from slot 1 (P1-04THM): channels 0..3
    float t0 = P1.readTemperature(1, 1);
    float t1 = P1.readTemperature(1, 2);
    // Read analog inputs from slot 2 (P1-4ADL2DAL-1): channels 0..1
    int a0 = P1.readAnalog(2, 1);
    int a1 = P1.readAnalog(2, 2);
    // Publish to Modbus so we can read them from outside
    modbusServer.holdingRegisterWrite(2, (uint16_t)(t0 * 10.0f));
    modbusServer.holdingRegisterWrite(3, (uint16_t)(t1 * 10.0f));
    modbusServer.holdingRegisterWrite(4, (uint16_t)a0);
    modbusServer.holdingRegisterWrite(5, (uint16_t)a1);
  }

  static unsigned long lastHb = 0;
  if (millis() - lastHb >= 1000) {
    lastHb = millis();
    Serial.print(F("alive t="));
    Serial.print(millis() / 1000);
    Serial.print(F("s, link="));
    Serial.println(Ethernet.linkStatus() == LinkON ? F("ON") : F("OFF/unk"));
  }
}
