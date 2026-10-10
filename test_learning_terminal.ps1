$ErrorActionPreference='Stop'
$Tokens=$null;$Errors=$null
$Ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'learning.ps1'),[ref]$Tokens,[ref]$Errors)
if($Errors.Count){throw $Errors[0]}
foreach($Name in @('Get-LinuxWorkerSnapshot','Invoke-LearningWorker')){
    $Function=$Ast.Find({param($Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq $Name},$true)
    if(-not $Function){throw "Missing worker function: $Name"}
    Invoke-Expression $Function.Extent.Text
}
$Focused=$true;$Continuous=$false;$Common=@();$WorkerScript='focused_experiment.py';$Session='test';$script:Calls=0
function wsl.exe {$global:LASTEXITCODE=0; $script:Calls++; return ('{"status":"'+$script:State+'","desired":"resume","worker_alive":false}')}
function Start-Process {throw 'Finished experiment reached process launch'}
foreach($State in @('completed','stopped','budget_exhausted')){
    $script:State=$State
    $Exit=Invoke-LearningWorker 'unused'
    if($Exit -ne 0){throw 'Finished experiment returned a failure'}
}
if($script:Calls -ne 3){throw 'Terminal preflight did not read current saved status'}
'PASS: periodic recovery checks never relaunch a finished finite experiment'
