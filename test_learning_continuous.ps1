$ErrorActionPreference='Stop'
$Tokens=$null;$Errors=$null
$Ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'learning.ps1'),[ref]$Tokens,[ref]$Errors)
if($Errors.Count){throw $Errors[0]}
foreach($Name in @('Format-LearningProgress','Get-SessionSwitches','Get-WorkerSettings','Get-WorkerTrigger','Test-LearningAwake')){
    $Function=$Ast.Find({param($Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq $Name},$true)
    if(-not $Function){throw "Missing continuous worker function: $Name"}
    Invoke-Expression $Function.Extent.Text
}
$Continuous=$true;$Research=$true;$FastReject=$true
if(@(Get-SessionSwitches -Linux).Count){throw 'Bounded trainer arguments leaked into supervisor'}
$script:Trigger=$null;$script:Settings=$null
function New-ScheduledTaskTrigger {param([switch]$AtLogOn,$User) $script:Trigger=@{AtLogOn=[bool]$AtLogOn;User=$User};return $script:Trigger}
function New-ScheduledTaskSettingsSet {param($ExecutionTimeLimit,[switch]$AllowStartIfOnBatteries,[switch]$DontStopIfGoingOnBatteries,$MultipleInstances,$RestartCount,$RestartInterval) $script:Settings=@{RestartCount=$RestartCount;RestartInterval=$RestartInterval;MultipleInstances=$MultipleInstances};return $script:Settings}
Get-WorkerTrigger 'test-user'|Out-Null;Get-WorkerSettings|Out-Null
if(-not $script:Trigger.AtLogOn -or $script:Trigger.User -ne 'test-user'){throw 'Login recovery must target the worker owner'}
if($script:Settings.RestartCount -ne 999 -or $script:Settings.MultipleInstances -ne 'IgnoreNew'){throw 'Task restart or duplicate protection missing'}
if(-not (Test-LearningAwake ([pscustomobject]@{status='waiting';desired='resume'}))){throw 'Automatic retries must keep the PC awake'}
foreach($Control in @('pause','stop')){
    if(Test-LearningAwake ([pscustomobject]@{status='waiting';desired=$Control})){throw 'User controls must release the keep-awake request'}
}
foreach($State in @('paused','blocked','stopped')){
    if(Test-LearningAwake ([pscustomobject]@{status=$State;desired='resume'})){throw 'Inactive worker must allow normal sleep'}
}
foreach($State in @('pausing','stopping')){
    if(-not (Test-LearningAwake ([pscustomobject]@{status=$State;desired='pause'}))){throw 'Checkpoint save must finish before normal idle sleep'}
}
$Value=[pscustomobject]@{continuous=$true;status='paused';phase='collect';round=3;active_seconds=28800;limit_seconds=$null;child_active_seconds=300;child_limit_seconds=43200;supervisor_detail='GPU released';detail='Paused';best_dev=@{tasks=@()}}
$Text=Format-LearningProgress $Value
if($Text -notlike '*until Pause or Stop*' -or $Text -like '*Active hours: 8 / 0*'){throw 'Continuous progress still presents a global time cutoff'}
$Continuous=$false
if(@(Get-WorkerTrigger 'test-user').Count){throw 'Finite experiments unexpectedly gained a login trigger'}
'PASS: continuous login recovery, bounded restarts and progress; finite mode unchanged'
