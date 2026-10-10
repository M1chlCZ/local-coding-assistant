$ErrorActionPreference='Stop'
$Tokens=$null;$Errors=$null
$Ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'learning.ps1'),[ref]$Tokens,[ref]$Errors)
if($Errors.Count){throw $Errors[0]}
foreach($Name in @('Get-WorkerTaskName','Write-WorkerStatus','Invoke-WorkerGuard')){
    $Function=$Ast.Find({param($Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq $Name},$true)
    if(-not $Function){throw "Missing durable worker diagnostic function: $Name"}
    Invoke-Expression $Function.Extent.Text
}
$ProjectRoot=Join-Path ([IO.Path]::GetTempPath()) ('learning-wrapper-'+[Guid]::NewGuid().ToString('N'))
$Session='.cache/learning/test-wrapper';$WorkerScript='focused_experiment.py'
$LogRoot=Join-Path $ProjectRoot '.cache/learning-windows';New-Item -ItemType Directory -Force $LogRoot|Out-Null
$LogPrefix=Join-Path $LogRoot 'attempt'
try {
    $Exit=Invoke-WorkerGuard $LogPrefix {throw 'Simulated Windows launch failure'}
    if($Exit -ne 1){throw 'Windows launch exception was reported as success'}
    $Saved=Get-Content (Join-Path $LogRoot ((Get-WorkerTaskName)+'.json')) -Raw|ConvertFrom-Json
    if($Saved.status -ne 'failed' -or $Saved.detail -notlike '*Simulated Windows launch failure*'){throw 'Startup failure was not saved for the panel'}
    if((Get-Content "$LogPrefix.wrapper.log" -Raw) -notlike '*Simulated Windows launch failure*'){throw 'Full startup exception was lost'}
    $Exit=Invoke-WorkerGuard $LogPrefix {75}
    if($Exit -ne 75){throw 'Transient native exit code was lost'}
    $Saved=Get-Content (Join-Path $LogRoot ((Get-WorkerTaskName)+'.json')) -Raw|ConvertFrom-Json
    if($Saved.status -ne 'failed' -or $Saved.exit_code -ne 75){throw 'Native worker failure was not saved'}
    $Exit=Invoke-WorkerGuard $LogPrefix {0}
    if($Exit -ne 0){throw 'Successful completion was reported as failure'}
    $Saved=Get-Content (Join-Path $LogRoot ((Get-WorkerTaskName)+'.json')) -Raw|ConvertFrom-Json
    if($Saved.status -ne 'exited' -or $Saved.exit_code -ne 0){throw 'Successful worker exit was not saved'}
    'PASS: startup exceptions and native exit codes are retained in durable diagnostics'
} finally {Remove-Item -Recurse -Force $ProjectRoot}
