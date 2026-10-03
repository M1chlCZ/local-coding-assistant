$ErrorActionPreference='Stop'
$Tokens=$null;$Errors=$null
$Ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'learning.ps1'),[ref]$Tokens,[ref]$Errors)
if($Errors.Count){throw $Errors[0]}
$Function=$Ast.Find({param($Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq 'Format-LearningProgress'},$true)
if(-not $Function){throw 'The Windows console has no live progress formatter.'}
Invoke-Expression $Function.Extent.Text
$State=[pscustomobject]@{status='running';phase='evaluate';round=3;active_seconds=3600;limit_seconds=43200;
    detail='Checking a coding task';best_dev=@{tasks=@(@{passed=$true},@{passed=$false})};
    research_score=17;research_regressions=1;research=$true;
    evaluation=@{completed=7;total=25;mode='adapter';checkpoint='checkpoint-10'}}
$Text=Format-LearningProgress $State
foreach($Expected in @('Round: 3','7/25','checkpoint-10','Retained: 1/2','Research: 17','lost: 1','1 / 12','Checking a coding task')){
    if(-not $Text.Contains($Expected)){throw ('Missing useful console progress: '+$Expected)}
}
$State.phase='train';$State|Add-Member -NotePropertyName training -NotePropertyValue @{step=3;max_steps=5}
if(-not (Format-LearningProgress $State).Contains('Training: 3/5')){throw 'Training steps are hidden.'}
$State.phase='collect';$State|Add-Member -NotePropertyName collected -NotePropertyValue 2;$State|Add-Member -NotePropertyName passed -NotePropertyValue 1
if(-not (Format-LearningProgress $State).Contains('Examples: 2 collected, 1 passed')){throw 'Collection progress is hidden.'}
'PASS: Windows console shows task counts, training steps, retained score, research score, and time limit'
