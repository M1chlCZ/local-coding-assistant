param([switch]$NoLaunch)
$ErrorActionPreference = 'Stop'
$Source = Join-Path $PSScriptRoot 'TrainingStudio.exe'
if (-not (Test-Path $Source)) { throw 'Extract the complete release ZIP before running install.ps1.' }
$Version = (Get-Item $Source).VersionInfo.ProductVersion.Split('+')[0]
if ($Version -notmatch '^\d+\.\d+\.\d+([.-][A-Za-z0-9.-]+)?$') { throw 'The app has an invalid version.' }
$Destination = Join-Path $env:LOCALAPPDATA ('TrainingStudio\app-' + $Version)
$Target = Join-Path $Destination 'TrainingStudio.exe'
$Running = Get-Process TrainingStudio -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $Target }
if ($Running) { throw 'This app version is running. Close it and pause its worker before reinstalling the same version.' }
New-Item -ItemType Directory -Force $Destination | Out-Null
Copy-Item $Source $Destination -Force
foreach ($Notice in @('LICENSE', 'licenses')) {
    $NoticePath = Join-Path $PSScriptRoot $Notice
    if (Test-Path $NoticePath) { Copy-Item $NoticePath $Destination -Recurse -Force }
}
# New versions install beside a running worker; only future launches use the new EXE.
Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {
    ($_.TaskName -like 'TrainingStudio-*' -and $_.Actions.Arguments -match '^--worker ') -or $_.TaskName -eq 'TrainingStudioUI'
} | ForEach-Object {
    $Task = $_
    foreach ($Action in $Task.Actions) {
        if ([IO.Path]::GetFileName($Action.Execute) -eq 'TrainingStudio.exe') { $Action.Execute = $Target }
    }
    Set-ScheduledTask -InputObject $Task | Out-Null
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
