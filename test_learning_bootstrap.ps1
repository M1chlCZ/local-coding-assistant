$ErrorActionPreference='Stop'
$Tokens=$null;$Errors=$null
$Ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'learning.ps1'),[ref]$Tokens,[ref]$Errors)
if($Errors.Count){throw $Errors[0]}
$Function=$Ast.Find({param($Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq 'Wait-LearningReady'},$true)
if(-not $Function){throw 'Worker starts without checking its WSL boot dependencies'}
Invoke-Expression $Function.Extent.Text
$Classifier=$Ast.Find({param($Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq 'Get-BootProbeResult'},$true)
if(-not $Classifier){throw 'A cold WSL service timeout is treated as a permanent configuration failure'}
Invoke-Expression $Classifier.Extent.Text
$Transient=Get-BootProbeResult 1 'Wsl/Service/CreateInstance/ERROR_TIMEOUT'
if($Transient.ExitCode -ne 75){throw 'Native WSL boot timeout was not marked retryable'}
$Fatal=Get-BootProbeResult 1 'Invalid controller configuration'
if($Fatal.ExitCode -ne 1){throw 'Unrelated configuration errors were hidden by boot retries'}
$Ready=Get-BootProbeResult 0 '{"ready":true,"reason":"Ready"}'
if($Ready.ExitCode -ne 0){throw 'Ready probe was rejected'}
$Continuous=$true;$Common=@('wsl-test');$Session='.cache/learning/continuous';$script:Calls=0
function Invoke-BootProbe {
    $script:Calls++
    if($script:Calls -eq 1){return [pscustomobject]@{ExitCode=75;Reason='Waiting for the model drive'}}
    return [pscustomobject]@{ExitCode=0;Reason='Ready'}
}
function Start-Sleep {param($Seconds)}
Wait-LearningReady 'test-log'
if($script:Calls -ne 2){throw 'Boot delay did not retry using a fresh WSL probe'}
$Continuous=$false;$script:Calls=0
Wait-LearningReady 'test-log'
if($script:Calls){throw 'Prepared bounded sessions unexpectedly changed their startup behavior'}
$Continuous=$true
function Invoke-BootProbe {return [pscustomobject]@{ExitCode=1;Reason='Invalid controller'}}
try {Wait-LearningReady 'test-log';throw 'Invalid configuration was retried instead of reported'}
catch {if($_.Exception.Message -notlike '*Invalid controller*'){throw}}
'PASS: fresh WSL readiness retries, fatal configuration reporting, finite-mode preservation'
