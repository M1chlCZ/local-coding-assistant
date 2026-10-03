$ErrorActionPreference='Stop'
$Tokens=$null;$Errors=$null
$Ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'learning.ps1'),[ref]$Tokens,[ref]$Errors)
if($Errors.Count){throw $Errors[0]}
$Function=$Ast.Find({param($Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq 'Get-SessionSwitches'},$true)
if(-not $Function){throw 'Windows controls cannot forward the fast rejection option.'}
Invoke-Expression $Function.Extent.Text
$Research=$true;$FastReject=$true
if((@(Get-SessionSwitches -Linux) -join ' ') -ne '--research --fast-reject'){throw 'The Linux worker did not receive both options.'}
if((@(Get-SessionSwitches) -join ' ') -ne '-Research -FastReject'){throw 'The Windows worker did not receive both options.'}
$Research=$false;$FastReject=$false
if(@(Get-SessionSwitches).Count){throw 'Existing sessions unexpectedly enabled research options.'}
'PASS: Windows controls forward session options and preserve existing defaults'
