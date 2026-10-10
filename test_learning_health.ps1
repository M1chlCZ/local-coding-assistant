$ErrorActionPreference='Stop'
$Tokens=$null;$Errors=$null
$Ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'learning.ps1'),[ref]$Tokens,[ref]$Errors)
if($Errors.Count){throw $Errors[0]}
foreach($Name in @('Get-WorkerTaskName','Add-WorkerHealth')){
    $Function=$Ast.Find({param($Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq $Name},$true)
    if(-not $Function){throw "Missing worker health function: $Name"}
    Invoke-Expression $Function.Extent.Text
}
$Session='.cache/learning/health-test';$ProjectRoot=$PSScriptRoot
$script:TaskState='Ready'
function Get-ScheduledTask {param($TaskName,$ErrorAction) [pscustomobject]@{State=$script:TaskState}}
$Value=[pscustomobject]@{status='running';phase='train';detail='Old training progress';worker_alive=$false;desired='resume'}
$Result=Add-WorkerHealth $Value
if($Result.status -ne 'interrupted' -or $Result.detail -notlike '*not running*'){throw 'Dead worker still appears active'}
if($Result.desired -ne 'resume'){throw 'Health display changed saved control'}
$script:TaskState='Running'
$Result=Add-WorkerHealth ([pscustomobject]@{status='interrupted';worker_alive=$false;desired='pause'})
if($Result.status -ne 'starting' -or $Result.desired -ne 'pause'){throw 'Windows startup lost saved pause or appeared failed'}
$Result=Add-WorkerHealth ([pscustomobject]@{status='running';desired='resume'})
if($Result.status -ne 'running'){throw 'Legacy finite worker without Linux health metadata lost its live progress'}
$script:TaskState='Ready'
$Result=Add-WorkerHealth ([pscustomobject]@{status='running';worker_alive=$true;desired='resume'})
if($Result.status -ne 'running'){throw 'A live Linux worker was called dead'}
foreach($State in @('paused','stopped','completed','blocked','failed')){
    $Result=Add-WorkerHealth ([pscustomobject]@{status=$State;worker_alive=$false;desired='pause'})
    if($Result.status -ne $State){throw 'Health display overwrote an inactive state'}
}
'PASS: task and process liveness replace stale active status without changing controls'
