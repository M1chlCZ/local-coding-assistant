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
$State|Add-Member -NotePropertyName recovered_examples -NotePropertyValue 1
$State|Add-Member -NotePropertyName training_retry -NotePropertyValue @{attempt=2;max_attempts=2}
$Text=Format-LearningProgress $State
if(-not $Text.Contains('Corrections recovered: 1') -or -not $Text.Contains('Training attempt: 2/2')){throw 'Failure correction progress is hidden.'}
$State|Add-Member -NotePropertyName continuous -NotePropertyValue $true
$State|Add-Member -NotePropertyName child_limit_seconds -NotePropertyValue 21600
$State|Add-Member -NotePropertyName benchmark -NotePropertyValue @{
    next_after_active_seconds=7200;
    last=@{base_passed=118;adapter_passed=121;total=164;gained=14;lost=11;reused=$false};
    progress=@{status='running';phase='adapter';completed=42;total=164}}
$Text=Format-LearningProgress $State
foreach($Expected in @('Next coding benchmark:','2 active hours','6-hour training maximum','Last coding benchmark: base 118/164; adapter 121/164','gained 14; lost 11','Coding benchmark:  adapter 42/164')){
    if(-not $Text.Contains($Expected)){throw ('Missing scheduled benchmark progress: '+$Expected)}
}
$State.benchmark.progress=$null;$State.benchmark.last.reused=$true
if(-not (Format-LearningProgress $State).Contains('unchanged weights; verified result reused')){throw 'Cached benchmark is presented as a new measurement.'}
$State.phase='confirmation'
$State|Add-Member -NotePropertyName confirmation_progress -NotePropertyValue @{mode='base';completed=12;total=20}
$Text=Format-LearningProgress $State
if(-not $Text.Contains('Fresh reserved checks: base 12/20')){throw 'Fresh confirmation progress is hidden behind completed training.'}
if($Text.Contains('Examples:')){throw 'Completed collection still appears active during confirmation.'}
'PASS: Windows console shows training, independent benchmark schedule, progress, complete scores, and reuse'
