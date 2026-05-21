# plc-disconnect.ps1
# Restore the Ethernet NIC to DHCP after working with the P1AM SCADA PLC.
#
# Removes any static IPs on the interface, switches it back to DHCP, and
# triggers a release/renew so Windows picks up an address from your router
# (or falls back to APIPA if nothing's listening, which is fine and means
# the NIC is back to its default behavior).
#
# Usage:
#   PS> .\plc-disconnect.ps1
#   PS> .\plc-disconnect.ps1 -InterfaceAlias 'Ethernet 2'

[CmdletBinding()]
param(
    [string]$InterfaceAlias = 'Ethernet'
)

# Self-elevate so the user doesn't have to open an admin shell.
if (-not ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Re-launching with admin rights (UAC prompt will appear)..." -ForegroundColor Yellow
    $args = @('-NoProfile','-ExecutionPolicy','Bypass','-File',$MyInvocation.MyCommand.Path,
              '-InterfaceAlias',$InterfaceAlias)
    Start-Process powershell -Verb RunAs -ArgumentList $args -Wait
    exit
}

Write-Host "=== PLC Disconnect ===" -ForegroundColor Cyan
Write-Host "Interface : $InterfaceAlias`n"

# Pull every static IPv4 address off the interface.
$existing = Get-NetIPAddress -InterfaceAlias $InterfaceAlias -AddressFamily IPv4 `
    -PrefixOrigin Manual -ErrorAction SilentlyContinue
foreach ($addr in $existing) {
    Write-Host "Removing static IP $($addr.IPAddress)/$($addr.PrefixLength)..."
    Remove-NetIPAddress -InterfaceAlias $InterfaceAlias -IPAddress $addr.IPAddress `
        -Confirm:$false -ErrorAction SilentlyContinue
}

# Switch the interface back to DHCP for IP and DNS.
Write-Host "Enabling DHCP for IP and DNS..."
Set-NetIPInterface -InterfaceAlias $InterfaceAlias -Dhcp Enabled -ErrorAction SilentlyContinue
Set-DnsClientServerAddress -InterfaceAlias $InterfaceAlias -ResetServerAddresses -ErrorAction SilentlyContinue

# Force a release/renew so the change takes effect immediately.
Write-Host "Releasing and renewing DHCP lease..."
ipconfig /release "$InterfaceAlias" 2>&1 | Out-Null
ipconfig /renew "$InterfaceAlias" 2>&1 | Out-Null

Start-Sleep -Seconds 2

# Report final state.
Write-Host "`nFinal address state:" -ForegroundColor Green
Get-NetIPAddress -InterfaceAlias $InterfaceAlias -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Format-Table InterfaceAlias, IPAddress, PrefixLength, PrefixOrigin -AutoSize
