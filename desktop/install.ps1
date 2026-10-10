param([switch]$NoLaunch)
$ErrorActionPreference = 'Stop'
$Destination = Join-Path $env:LOCALAPPDATA 'TrainingStudio\app'
$Source = Join-Path $PSScriptRoot 'TrainingStudio.exe'
if (-not (Test-Path $Source)) { throw 'Extract the complete release ZIP before running install.ps1.' }
$Running = Get-Process TrainingStudio -ErrorAction SilentlyContinue
if ($Running) { throw 'Close Training Studio and pause any Studio background worker before replacing the app. Do not interrupt a saving checkpoint.' }
New-Item -ItemType Directory -Force $Destination | Out-Null
Copy-Item $Source $Destination -Force
foreach ($Notice in @('LICENSE', 'licenses')) {
    $NoticePath = Join-Path $PSScriptRoot $Notice
    if (Test-Path $NoticePath) { Copy-Item $NoticePath $Destination -Recurse -Force }
}
$Shell = New-Object -ComObject WScript.Shell
foreach ($Folder in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
    $Shortcut = $Shell.CreateShortcut((Join-Path $Folder 'Training Studio.lnk'))
    $Shortcut.TargetPath = Join-Path $Destination 'TrainingStudio.exe'
    $Shortcut.WorkingDirectory = $Destination
    $Shortcut.Description = 'Local CUDA fine-tuning and measured coding results'
    $Shortcut.Save()
}
if (-not $NoLaunch) { Start-Process (Join-Path $Destination 'TrainingStudio.exe') }
Write-Host 'Training Studio is installed for this Windows user.'
