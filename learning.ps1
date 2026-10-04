param(
    [ValidateSet('panel','start','pause','resume','stop','status','worker','watch')][string]$Action = 'panel',
    [double]$Hours = 12,
    [string]$Session = '.cache/learning/tuned',
    [switch]$Research,
    [switch]$FastReject,
    [switch]$Continuous
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LinuxRoot = '/home/coder/local-coding-assistant'
$ModelPath = Join-Path $ProjectRoot '.cache\models\Qwen3.8-27B-UD-Q4_K_M.gguf'
$Common = @('-d','LocalCodingAssistant','-u','coder','--cd',$LinuxRoot,'--exec')
$env:WSL_UTF8 = '1'
if ($Continuous -and -not $PSBoundParameters.ContainsKey('Session')) { $Session = '.cache/learning/continuous' }
$WorkerScript = if ($Continuous) { 'continuous_learning.py' } else { 'learning_session.py' }

function Join-NativeArguments($Items) {
    if ($Items | Where-Object { $_ -match '["\r\n]' }) { throw 'Invalid argument character.' }
    return (($Items | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join ' ')
}

function Get-SessionSwitches([switch]$Linux) {
    if ($Continuous) { if (-not $Linux) { '-Continuous' }; return }
    if ($Research) { if ($Linux) { '--research' } else { '-Research' } }
    if ($FastReject) { if ($Linux) { '--fast-reject' } else { '-FastReject' } }
}

function Get-WorkerSettings {
    $Options = @{ ExecutionTimeLimit=[TimeSpan]::Zero; AllowStartIfOnBatteries=$true;
        DontStopIfGoingOnBatteries=$true; MultipleInstances='IgnoreNew' }
    if ($Continuous) { $Options.RestartCount=3; $Options.RestartInterval=[TimeSpan]::FromMinutes(1) }
    return New-ScheduledTaskSettingsSet @Options
}

function Get-WorkerTrigger([string]$Owner) {
    if ($Continuous) { return New-ScheduledTaskTrigger -AtLogOn -User $Owner }
}

function Get-BootProbeResult([int]$ExitCode,[string]$Output) {
    $Reason = 'WSL readiness probe failed; see boot logs'
    try {
        $Value = $Output | ConvertFrom-Json
        if ($Value.reason) { $Reason=$Value.reason }
    } catch {
        if ($ExitCode -ne 0 -and $Output -match 'Wsl/Service/CreateInstance/ERROR_TIMEOUT') {
            return [pscustomobject]@{ ExitCode=75; Reason='Waiting for the WSL service to finish starting' }
        }
    }
    return [pscustomobject]@{ ExitCode=$ExitCode; Reason=$Reason }
}

function Invoke-BootProbe([string]$LogPrefix) {
    $Items = $Common + @('/usr/bin/python3','learning_bootstrap.py','--session',$Session,'--continuous')
    $Out = "$LogPrefix.boot.out.log"; $Err = "$LogPrefix.boot.err.log"
    # Keep native WSL stderr as data. Early boot warnings must not become PowerShell exceptions.
    $Probe = Start-Process 'wsl.exe' -ArgumentList (Join-NativeArguments $Items) -WindowStyle Hidden `
        -PassThru -RedirectStandardOutput $Out -RedirectStandardError $Err
    $Handle = $Probe.Handle
    $Probe.WaitForExit(); $Probe.Refresh()
    return Get-BootProbeResult $Probe.ExitCode (Get-Content $Out -Raw)
}

function Wait-LearningReady([string]$LogPrefix) {
    if (-not $Continuous) { return }
    $Deadline = (Get-Date).AddMinutes(2)
    do {
        # A new invocation gets a fresh mount namespace; one long-lived failed view cannot recover itself.
        $Probe = Invoke-BootProbe $LogPrefix
        if ($Probe.ExitCode -eq 0) { return }
        if ($Probe.ExitCode -ne 75) { throw $Probe.Reason }
        Write-Host $Probe.Reason
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $Deadline)
    throw 'WSL boot dependencies remain unavailable; Windows will retry the worker. See boot logs.'
}

if ($Action -eq 'worker') {
    $LogRoot = Join-Path $ProjectRoot '.cache\learning-windows'
    New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
    $Stamp = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-ffff')
    Add-Type @'
using System.Runtime.InteropServices;
public static class LearningPower {
    [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
}
'@
    Wait-LearningReady (Join-Path $LogRoot $Stamp)
    $NativeArguments = $Common + @('.cache/rlm-env/bin/python',$WorkerScript,'run','--session',$Session)
    if (-not $Continuous) { $NativeArguments += @('--hours',$Hours.ToString([Globalization.CultureInfo]::InvariantCulture),'--model',$ModelPath) }
    $NativeArguments += @(Get-SessionSwitches -Linux)
    $WorkerProcess = Start-Process 'wsl.exe' -ArgumentList (Join-NativeArguments $NativeArguments) `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $LogRoot "$Stamp.out.log") `
        -RedirectStandardError (Join-Path $LogRoot "$Stamp.err.log")
    # Keep a process handle before exit so Windows PowerShell retains the native exit code.
    $WorkerHandle = $WorkerProcess.Handle
    try {
        while (-not $WorkerProcess.HasExited) {
            try {
                $Snapshot = & wsl.exe @Common '.cache/rlm-env/bin/python' $WorkerScript 'status' '--session' $Session | ConvertFrom-Json
                $Paused = $Snapshot.status -in @('paused','blocked','waiting','stopped')
            } catch { $Paused = $false }
            # Keep the PC awake during work; pause restores normal idle sleep. Display sleep stays allowed.
            $Flags = if ($Paused) { [uint32]2147483648 } else { [uint32]2147483649 }
            [void][LearningPower]::SetThreadExecutionState($Flags)
            Start-Sleep -Seconds 5
            $WorkerProcess.Refresh()
        }
        $WorkerProcess.WaitForExit()
        exit $WorkerProcess.ExitCode
    } finally { [void][LearningPower]::SetThreadExecutionState([uint32]2147483648) }
}

function Invoke-Control([string]$Command) {
    if ($Continuous -and $Command -eq 'status') {
        $Result = & wsl.exe @Common '.cache/rlm-env/bin/python' 'learning_progress.py' '--session' $Session
    } else {
        $Result = & wsl.exe @Common '.cache/rlm-env/bin/python' $WorkerScript $Command '--session' $Session
    }
    if ($LASTEXITCODE -ne 0) { throw "Session command failed: $Result" }
    return ($Result -join "`n")
}

function Format-LearningProgress($Value) {
    $Used = [Math]::Round($Value.active_seconds/3600,2)
    $Limit = [Math]::Round($Value.limit_seconds/3600,2)
    $Lines = @("State: $($Value.status)  Round: $($Value.round)  Phase: $($Value.phase)", "Active hours: $Used / $Limit")
    if ($Value.continuous) {
        $Lines[1] = "Total training hours: $Used; runs until Pause or Stop"
        $Lines += "$($Value.supervisor_detail)"
        if ($Value.status -eq 'paused') { $Lines += 'GPU released for gaming. Click Resume when ready.' }
        if ($Value.benchmark) {
            $Remaining = [Math]::Round($Value.benchmark.next_after_active_seconds/3600,2)
            $Lines += "Next HumanEval audit: after this experiment; about $Remaining active hours remain (12-hour maximum)."
            $Last = $Value.benchmark.last
            if ($Last) {
                $Lines += "Last HumanEval: base $($Last.base_passed)/$($Last.total); adapter $($Last.adapter_passed)/$($Last.total); gained $($Last.gained); lost $($Last.lost)"
                if ($Last.reused) { $Lines += 'HumanEval: unchanged weights; verified result reused.' }
            }
            $Audit = $Value.benchmark.progress
            if ($Audit) { $Lines += "HumanEval: $($Audit.phase) $($Audit.completed)/$($Audit.total) ($($Audit.status))" }
        }
    }
    $Tasks = @($Value.best_dev.tasks | Where-Object { $null -ne $_ })
    if ($Tasks.Count) {
        $Passed = @($Tasks | Where-Object { $_.passed -eq $true }).Count
        $Lines += "Retained: $Passed/$($Tasks.Count)"
        if ($Value.research -and $null -ne $Value.research_score) {
            $Lines += "Research: $($Value.research_score)/$($Tasks.Count)  lost: $($Value.research_regressions)"
        }
    }
    switch ($Value.phase) {
        'collect' { $Lines += "Examples: $($Value.collected) collected, $($Value.passed) passed" }
        'train' { $Lines += "Training: $($Value.training.step)/$($Value.training.max_steps)" }
        'evaluate' { $Lines += "Tests: $($Value.evaluation.completed)/$($Value.evaluation.total)  $($Value.evaluation.mode) $($Value.evaluation.checkpoint)" }
        'confirmation' { $Lines += "Fresh reserved checks: $($Value.confirmation_progress.mode) $($Value.confirmation_progress.completed)/$($Value.confirmation_progress.total)" }
    }
    $Lines += $Value.detail
    return ($Lines -join "`r`n")
}

function Watch-LearningProgress {
    Write-Host 'Live coding research progress. Closing this window keeps the worker running.'
    $Previous = ''; $Printed = [DateTime]::MinValue
    while ($true) {
        $Value = Invoke-Control 'status' | ConvertFrom-Json
        if (-not $Value.status) { throw 'No prepared learning session exists at this location.' }
        $Text = Format-LearningProgress $Value
        if ($Text -ne $Previous -or ((Get-Date)-$Printed).TotalSeconds -ge 30) {
            Write-Host ("`n"+(Get-Date).ToString('HH:mm:ss')+"`n"+$Text)
            $Previous = $Text; $Printed = Get-Date
        }
        if ($Value.status -in @('completed','stopped','failed')) { return }
        Start-Sleep -Seconds 5
    }
}

function Grant-TaskControl([string]$TaskName) {
    # SSH can create the task with an elevated token; the desktop panel uses a normal token.
    $Service = New-Object -ComObject 'Schedule.Service'
    $Service.Connect()
    $Task = $Service.GetFolder('\').GetTask($TaskName)
    $Sid = [Security.Principal.WindowsIdentity]::GetCurrent().User
    $Descriptor = New-Object Security.AccessControl.RawSecurityDescriptor($Task.GetSecurityDescriptor(7))
    foreach ($ExistingAce in $Descriptor.DiscretionaryAcl) {
        if ($ExistingAce.SecurityIdentifier -eq $Sid -and $ExistingAce.AceQualifier -eq 'AccessAllowed' -and
            ($ExistingAce.AccessMask -band 0x1f01ff) -eq 0x1f01ff) { return }
    }
    $Ace = New-Object Security.AccessControl.CommonAce([Security.AccessControl.AceFlags]::None,
        [Security.AccessControl.AceQualifier]::AccessAllowed,0x1f01ff,$Sid,$false,$null)
    $Descriptor.DiscretionaryAcl.InsertAce(0,$Ace)
    $Task.SetSecurityDescriptor($Descriptor.GetSddlForm('All'),0)
}

function Start-Worker {
    if ($Hours -lt 0.01 -or $Hours -gt 24) { throw 'Use 0.01 to 24 hours.' }
    $Saved = Invoke-Control 'status' | ConvertFrom-Json
    if ($Saved.status -in @('completed','stopped')) {
        throw 'This session is finished. Choose a new session folder for fresh training tasks.'
    }
    Invoke-Control 'resume' | Out-Null
    # An on-demand Task Scheduler job survives OpenSSH's child-process cleanup.
    # Continuous recovery uses login plus the persisted command, without a stored password.
    $WorkerArguments = @('-NoProfile','-ExecutionPolicy','Bypass','-File',
        (Join-Path $ProjectRoot 'learning.ps1'),'-Action','worker','-Session',$Session,'-Hours',
        $Hours.ToString([Globalization.CultureInfo]::InvariantCulture))
    $WorkerArguments += @(Get-SessionSwitches)
    $ArgumentText = Join-NativeArguments $WorkerArguments
    $Hash = [Security.Cryptography.SHA256]::Create()
    try { $Suffix = ([BitConverter]::ToString($Hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($Session)))).Replace('-','').Substring(0,8) }
    finally { $Hash.Dispose() }
    $TaskName = "LocalCodingAssistantLearning-$Suffix"
    $Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($Existing -and $Existing.State -eq 'Running') { return Invoke-Control 'status' }
    $TaskAction = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -Argument $ArgumentText -WorkingDirectory $ProjectRoot
    $Principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
        -LogonType Interactive -RunLevel Limited
    $Settings = Get-WorkerSettings
    $Registration = @{ TaskName=$TaskName; Action=$TaskAction; Principal=$Principal; Settings=$Settings; Force=$true }
    $Trigger = Get-WorkerTrigger ([Security.Principal.WindowsIdentity]::GetCurrent().Name)
    if ($Trigger) { $Registration.Trigger=$Trigger }
    Register-ScheduledTask @Registration | Out-Null
    Grant-TaskControl $TaskName
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 2
    if ((Get-ScheduledTask -TaskName $TaskName).State -ne 'Running') {
        $Info = Get-ScheduledTaskInfo -TaskName $TaskName
        $Final = Invoke-Control 'status' | ConvertFrom-Json
        if ($Info.LastTaskResult -eq 0 -and $Final.status -in @('completed','stopped')) {
            return Invoke-Control 'status'
        }
        if ($Final.status -eq 'failed') { throw "Worker failed: $($Final.detail)" }
        throw "Worker did not start. Task result: $($Info.LastTaskResult). See .cache\learning-windows."
    }
    return Invoke-Control 'status'
}

switch ($Action) {
    'start' { Start-Worker; exit }
    'resume' { Start-Worker; exit }
    'pause' { Invoke-Control 'pause'; exit }
    'stop' { Invoke-Control 'stop'; exit }
    'status' { Invoke-Control 'status'; exit }
    'watch' { Watch-LearningProgress; return }
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$Form = New-Object Windows.Forms.Form
$Form.Text = 'Local Coding Assistant - Learning'
$Form.Size = New-Object Drawing.Size(680,360)
$Form.MinimumSize = $Form.Size
$Form.StartPosition = 'CenterScreen'
$Label = New-Object Windows.Forms.Label
$Label.Location = New-Object Drawing.Point(15,15)
$Label.Size = New-Object Drawing.Size(635,50)
$Label.Text = if ($Continuous) { 'Continuous CUDA learning. Pause saves progress and releases the GPU for gaming. Resume continues it. Closing this panel keeps learning running.' } else { "CUDA training on this PC. Limit: $Hours active hours.`nPause saves progress and frees the GPU. Closing this panel keeps the session running." }
$Form.Controls.Add($Label)
$Status = New-Object Windows.Forms.TextBox
$Status.Multiline = $true
$Status.ReadOnly = $true
$Status.ScrollBars = 'Vertical'
$Status.Location = New-Object Drawing.Point(15,110)
$Status.Size = New-Object Drawing.Size(635,195)
$Status.Anchor = 'Top,Bottom,Left,Right'
$Form.Controls.Add($Status)
$Buttons = @()
$i=0
foreach ($Name in @('Start','Pause','Resume','Stop','Refresh')) {
    $Button = New-Object Windows.Forms.Button
    $Button.Text = $Name
    $Button.Tag = $Name.ToLowerInvariant()
    $Button.Location = New-Object Drawing.Point((15+125*$i),70)
    $Button.Size = New-Object Drawing.Size(115,30)
    $Button.Add_Click({
        param($Sender,$Event)
        try {
            $Form.UseWaitCursor = $true
            switch ($Sender.Tag) {
                'start' { Start-Worker | Out-Null }
                'resume' { Start-Worker | Out-Null }
                'refresh' { }
                default { Invoke-Control $Sender.Tag | Out-Null }
            }
            $Status.Text = Format-LearningProgress (Invoke-Control 'status' | ConvertFrom-Json)
        } catch { $Status.Text = $_.Exception.Message }
        finally { $Form.UseWaitCursor = $false }
    })
    $Form.Controls.Add($Button)
    $Buttons += $Button
    $i++
}
$Timer = New-Object Windows.Forms.Timer
$Timer.Interval = 5000
$script:LastProgress = ''
$Timer.Add_Tick({
    try {
        $Value = Invoke-Control 'status' | ConvertFrom-Json
        $Status.Text = Format-LearningProgress $Value
        if ($Status.Text -ne $script:LastProgress) {
            Write-Host ("`n"+(Get-Date).ToString('HH:mm:ss')+"`n"+$Status.Text)
            $script:LastProgress = $Status.Text
        }
    } catch { $Status.Text = $_.Exception.Message }
})
$Timer.Start()
try { [void]$Form.ShowDialog() } finally { $Timer.Stop(); $Timer.Dispose(); $Form.Dispose() }
