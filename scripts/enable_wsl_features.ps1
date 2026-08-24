#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'

dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
if ($LASTEXITCODE -ne 0) { throw "WSL feature enablement failed: $LASTEXITCODE" }

dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
if ($LASTEXITCODE -ne 0) { throw "Virtual Machine Platform enablement failed: $LASTEXITCODE" }

Write-Host 'WSL features enabled. Restart Windows, then run: wsl --install -d Ubuntu'
