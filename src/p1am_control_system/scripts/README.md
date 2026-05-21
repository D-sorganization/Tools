# P1AM PLC connection scripts

The P1AM-100 SCADA PLC lives on an **isolated subnet** (`192.168.1.0/24`)
for industrial safety/predictability. Your PC's Ethernet NIC needs a static
address on that subnet to talk to it. These scripts automate the toggle
without putting the PLC on your main LAN.

## Usage

```powershell
# Before working with the PLC (sets NIC to static 192.168.1.50/24, no gateway)
.\plc-connect.ps1

# When you're done (restores DHCP on the Ethernet NIC)
.\plc-disconnect.ps1
```

Each script self-elevates — you'll get a UAC prompt once per run.

If your Ethernet adapter has a name other than `Ethernet` (check
`Get-NetAdapter | Where-Object Status -eq 'Up'`), pass it explicitly:

```powershell
.\plc-connect.ps1 -InterfaceAlias 'Ethernet 2'
.\plc-disconnect.ps1 -InterfaceAlias 'Ethernet 2'
```

## Why this design

- **Static IP for the PLC, isolated subnet** — SCADA-network best practice.
  Process traffic and general-purpose LAN traffic don't mix.
- **No default gateway in the static config** — preserves your main LAN's
  default route (typically on Wi-Fi at `192.168.4.1` or similar). Internet
  traffic keeps working while the PLC link is up.
- **Self-elevation** — Windows requires admin to change IP config. The
  scripts re-launch themselves under UAC so you don't have to remember to
  open an admin shell.
- **No registry hackery, no driver fiddling** — pure `Get-NetIPAddress` /
  `New-NetIPAddress` / `Set-NetIPInterface`. Reverts cleanly.

## Quick verify

After `plc-connect.ps1`, you should be able to:

```powershell
ping 192.168.1.100              # should reply with 1-2 ms RTT
python -c "from pymodbus.client import ModbusTcpClient as M; c=M('192.168.1.100',port=502); print(c.connect())"
```

If ping fails, check the Cat 6 cable, that the PLC's 24 VDC supply is on,
and that the PLC's PWR LED is solid.
