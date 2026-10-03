$ErrorActionPreference='Stop'
$Tokens=$null;$Errors=$null
$Ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'learning.ps1'),[ref]$Tokens,[ref]$Errors)
if($Errors.Count){throw $Errors[0]}
foreach($Name in @('Start-Worker','Join-NativeArguments','Get-SessionSwitches','Get-WorkerSettings','Get-WorkerTrigger')){
    $Function=$Ast.Find({param($Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq $Name},$true)
    Invoke-Expression $Function.Extent.Text
}
$Hours=12;$Session='.cache/learning/test';$ProjectRoot=$PSScriptRoot;$Research=$true;$FastReject=$true
$script:Controls=@();$script:Registered=$false;$script:FakeState=[pscustomobject]@{status='completed';detail='No training tasks remain'}
function Invoke-Control($Command){$script:Controls+=@($Command);return ($script:FakeState|ConvertTo-Json)}
function Get-ScheduledTask {param($TaskName,$ErrorAction) return [pscustomobject]@{State='Ready'}}
function Get-ScheduledTaskInfo {param($TaskName) return [pscustomobject]@{LastTaskResult=0}}
function New-ScheduledTaskAction {param($Execute,$Argument,$WorkingDirectory)}
function New-ScheduledTaskPrincipal {param($UserId,$LogonType,$RunLevel)}
function New-ScheduledTaskSettingsSet {param($ExecutionTimeLimit,[switch]$AllowStartIfOnBatteries,[switch]$DontStopIfGoingOnBatteries,$MultipleInstances)}
function Register-ScheduledTask {param($TaskName,$Action,$Principal,$Settings,$Trigger,[switch]$Force) $script:Registered=$true}
function Grant-TaskControl($TaskName){}
function Start-ScheduledTask {param($TaskName) $script:FakeState.status='completed'}
function Start-Sleep {param($Seconds)}
foreach($Finished in @('completed','stopped')){
    $script:FakeState.status=$Finished;$script:Controls=@();$script:Registered=$false
    try {Start-Worker|Out-Null;throw 'Finished session was not rejected'}
    catch {if($_.Exception.Message -notlike '*finished*'){throw}}
    if($script:Registered -or 'resume' -in $script:Controls){throw 'A finished session was restarted'}
}
$script:FakeState.status='paused';$script:Registered=$false
$Result=Start-Worker|ConvertFrom-Json
if($Result.status -ne 'completed' -or -not $script:Registered){throw 'Fast completion was reported as a launch failure'}
'PASS: start preserves finished sessions and accepts a worker that completes during startup'
