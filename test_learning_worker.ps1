param([string]$ProjectRoot = $PSScriptRoot)
$ErrorActionPreference='Stop';$env:WSL_UTF8='1'
$Suffix=[Guid]::NewGuid().ToString('N')
$Session='.cache/learning/worker-exit-test-'+$Suffix
$TaskName='LocalCodingAssistantExitTest-'+$Suffix
$Common=@('-d','LocalCodingAssistant','-u','coder','--cd','/home/coder/local-coding-assistant','--exec')
# Invalid task binding makes Python exit before any model can start.
$Prepare="import json;from pathlib import Path;p=Path('$Session');p.mkdir(parents=True);(p/'status.json').write_text(json.dumps(dict(status='failed',phase='collect',round=1,active_seconds=0,limit_seconds=36,tasks_sha256='invalid',sources={},detail='Deliberately invalid worker test')))"
& wsl.exe @Common python3 -c $Prepare
if($LASTEXITCODE -ne 0){throw 'Test session setup failed.'}
try {
    $Action=New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument ('-NoProfile -ExecutionPolicy Bypass -File "'+(Join-Path $ProjectRoot 'learning.ps1')+'" -Action worker -Session '+$Session+' -Hours 0.01') -WorkingDirectory $ProjectRoot
    $Principal=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $Settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 2) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Principal $Principal -Settings $Settings -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
    $Deadline=(Get-Date).AddSeconds(60)
    do { Start-Sleep -Milliseconds 500;$Task=Get-ScheduledTask -TaskName $TaskName;$Info=Get-ScheduledTaskInfo -TaskName $TaskName }
    while (($Task.State -eq 'Running' -or $Info.LastRunTime.Year -lt 2000) -and (Get-Date) -lt $Deadline)
    if($Task.State -eq 'Running'){throw 'Worker exit test timed out.'}
    if($Info.LastTaskResult -ne 1){throw "A failed Python worker was reported as task result $($Info.LastTaskResult), expected 1."}
    'PASS: Windows reports a failed Python worker as a failed task'
} finally {
    $Task=Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if($Task){if($Task.State -eq 'Running'){Stop-ScheduledTask -TaskName $TaskName};Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false}
    & wsl.exe @Common python3 -c "import shutil;shutil.rmtree('$Session')"
}
