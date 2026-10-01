param([string]$TaskName = 'LocalCodingAssistantLearning-8D2767D7')
$ErrorActionPreference = 'Stop'
$Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$Principal = New-Object Security.Principal.WindowsPrincipal($Identity)
if ($Principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this check from a normal Windows terminal, without administrator elevation.'
}
$Tokens = $null; $Errors = $null
$Ast = [Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $PSScriptRoot 'learning.ps1'), [ref]$Tokens, [ref]$Errors)
if ($Errors.Count) { throw $Errors[0] }
$Function = $Ast.Find({ param($Node)
    $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq 'Grant-TaskControl'
}, $true)
Invoke-Expression $Function.Extent.Text
$Service = New-Object -ComObject 'Schedule.Service'; $Service.Connect()
$Task = $Service.GetFolder('\').GetTask($TaskName)
Register-ScheduledTask -TaskName $TaskName -Xml $Task.Xml -Force | Out-Null
Grant-TaskControl $TaskName
$Before = $Task.GetSecurityDescriptor(7)
Grant-TaskControl $TaskName
if ($Task.GetSecurityDescriptor(7) -ne $Before) { throw 'Repeated permission setup changed the task ACL.' }
'PASS: normal desktop login can update the task; permission setup is idempotent.'
