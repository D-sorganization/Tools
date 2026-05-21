# plc-connect.ps1
# Configure the Ethernet NIC to talk to the P1AM-100 SCADA PLC at 192.168.1.100.
#
# The PLC lives on an isolated 192.168.1.0/24 subnet (industrial practice).
# Your Ethernet NIC needs a static address on that subnet to reach it. This
# script sets 192.168.1.50/24 with no gateway (preserves your main LAN's
# default route, which lives on Wi-Fi).
#
# Usage:
#   PS> .\plc-connect.ps1
#   PS> .\plc-connect.ps1 -InterfaceAlias 'Ethernet 2'   # if your NIC has a different name
#
# Re-run plc-disconnect.ps1 to switch the NIC back to DHCP.

[CmdletBinding()]
param(
    [string]$InterfaceAlias = 'Ethernet',
    [string]$IPAddress      = '192.168.1.50',
    [int]$PrefixLength      = 24,
    [string]$PlcAddress     = '192.168.1.100'
)

# Self-elevate so the user doesn't have to remember to open admin shells.
if (-not ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Re-launching with admin rights (UAC prompt will appear)..." -ForegroundColor Yellow
    $args = @('-NoProfile','-ExecutionPolicy','Bypass','-File',$MyInvocation.MyCommand.Path,
              '-InterfaceAlias',$InterfaceAlias,'-IPAddress',$IPAddress,
              '-PrefixLength',$PrefixLength,'-PlcAddress',$PlcAddress)
    Start-Process powershell -Verb RunAs -ArgumentList $args -Wait
    exit
}

Write-Host "=== PLC Connect ===" -ForegroundColor Cyan
Write-Host "Interface : $InterfaceAlias"
Write-Host "Setting   : $IPAddress/$PrefixLength (static, no gateway)"
Write-Host "Target PLC: $PlcAddress`n"

# Remove any existing static IP on this interface so we don't accumulate them.
$existing = Get-NetIPAddress -InterfaceAlias $InterfaceAlias -AddressFamily IPv4 `
    -PrefixOrigin Manual -ErrorAction SilentlyContinue
foreach ($addr in $existing) {
    Write-Host "Removing existing static IP $($addr.IPAddress)/$($addr.PrefixLength)..."
    Remove-NetIPAddress -InterfaceAlias $InterfaceAlias -IPAddress $addr.IPAddress `
        -Confirm:$false -ErrorAction SilentlyContinue
}

# Add the new static IP.
try {
    New-NetIPAddress -InterfaceAlias $InterfaceAlias -IPAddress $IPAddress `
        -PrefixLength $PrefixLength -ErrorAction Stop | Out-Null
    Write-Host "Static IP applied: $IPAddress/$PrefixLength" -ForegroundColor Green
} catch {
    Write-Host "Failed to set static IP: $_" -ForegroundColor Red
    exit 1
}

# Brief settle so Windows binds the address before we ping.
Start-Sleep -Seconds 2

# Verify by pinging the PLC.
Write-Host "`nPinging PLC at $PlcAddress..."
$result = Test-Connection -ComputerName $PlcAddress -Count 3 -Quiet
if ($result) {
    Write-Host "PLC is reachable. You can now talk to it (port 502 for Modbus TCP)." -ForegroundColor Green
} else {
    Write-Host "PLC did not respond. Check Cat 6 cable + 24V supply + PLC power LED." -ForegroundColor Yellow
}

Write-Host "`nRun plc-disconnect.ps1 when you're done to restore DHCP."
