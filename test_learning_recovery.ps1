param([string]$ProjectRoot = $PSScriptRoot)
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$Tokens=$null;$Errors=$null
$Ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $ProjectRoot 'learning.ps1'),[ref]$Tokens,[ref]$Errors)
if($Errors.Count){throw $Errors[0]}
foreach($Name in @('Get-WorkerSettings','Get-WorkerTrigger')){
    $Function=$Ast.Find({param($Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq $Name},$true)
    Invoke-Expression $Function.Extent.Text
}
$Focused=$true;$Continuous=$false
$Suffix=[Guid]::NewGuid().ToString('N');$TaskName='LocalCodingAssistantRecoveryTest-'+$Suffix
$Directory=Join-Path ([IO.Path]::GetTempPath()) $TaskName;New-Item -ItemType Directory $Directory|Out-Null
$Count=Join-Path $Directory 'count';$Script=Join-Path $Directory 'exit-probe.ps1'
$Body='$n=0;if(Test-Path "'+$Count+'"){$n=[int](Get-Content "'+$Count+'")};$n++;Set-Content "'+$Count+'" $n;if($n -eq 1){exit 75};exit 0'
Set-Content $Script $Body
try {
    $Owner=[Security.Principal.WindowsIdentity]::GetCurrent().Name
    $Triggers=@(Get-WorkerTrigger $Owner)
    if($Triggers.Count -ne 2){throw 'No independent recovery trigger exists'}
    # Keep the production repetition policy; bring only the first recovery tick forward for this test.
    $Triggers[1].StartBoundary=(Get-Date).AddSeconds(20).ToString('s')
    $Action=New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument ('-NoProfile -ExecutionPolicy Bypass -File "'+$Script+'"')
    $Principal=New-ScheduledTaskPrincipal -UserId $Owner -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Principal $Principal -Settings (Get-WorkerSettings) -Trigger $Triggers -Force|Out-Null
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 4
    if((Get-ScheduledTaskInfo -TaskName $TaskName).LastTaskResult -ne 75){throw 'Probe did not reproduce a nonzero application exit'}
    $Deadline=(Get-Date).AddSeconds(45)
    do {Start-Sleep -Seconds 1;$Attempts=if(Test-Path $Count){[int](Get-Content $Count)}else{0}}
    while($Attempts -lt 2 -and (Get-Date) -lt $Deadline)
    if($Attempts -ne 2){throw "Recovery trigger did not relaunch the failed application: $Attempts attempts"}
    Start-Sleep -Seconds 2
    if((Get-ScheduledTaskInfo -TaskName $TaskName).LastTaskResult -ne 0){throw 'Recovered application did not finish successfully'}
    'PASS: Windows recovery trigger relaunches a nonzero application exit without manual intervention'
} finally {
    $Task=Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if($Task){if($Task.State -eq 'Running'){Stop-ScheduledTask -TaskName $TaskName};Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false}
    Remove-Item -Recurse -Force $Directory
}
