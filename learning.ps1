param(
    [ValidateSet('panel','start','pause','resume','stop','status','worker')][string]$Action = 'panel',
    [double]$Hours = 12,
    [string]$Session = '.cache/learning/current'
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LinuxRoot = '/home/coder/local-coding-assistant'
$ModelPath = Join-Path $ProjectRoot '.cache\models\Qwen3.8-27B-UD-Q4_K_M.gguf'
$Common = @('-d','LocalCodingAssistant','-u','coder','--cd',$LinuxRoot,'--exec')
$env:WSL_UTF8 = '1'

function Join-NativeArguments($Items) {
    if ($Items | Where-Object { $_ -match '["\r\n]' }) { throw 'Invalid argument character.' }
    return (($Items | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join ' ')
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
    $NativeArguments = $Common + @('.cache/rlm-env/bin/python','learning_session.py','run','--session',$Session,
        '--hours',$Hours.ToString([Globalization.CultureInfo]::InvariantCulture),'--model',$ModelPath)
    $WorkerProcess = Start-Process 'wsl.exe' -ArgumentList (Join-NativeArguments $NativeArguments) `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $LogRoot "$Stamp.out.log") `
        -RedirectStandardError (Join-Path $LogRoot "$Stamp.err.log")
    try {
        while (-not $WorkerProcess.HasExited) {
            try {
                $Snapshot = & wsl.exe @Common '.cache/rlm-env/bin/python' 'learning_session.py' 'status' '--session' $Session | ConvertFrom-Json
                $Paused = $Snapshot.status -eq 'paused'
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
    $Result = & wsl.exe @Common '.cache/rlm-env/bin/python' 'learning_session.py' $Command '--session' $Session
    if ($LASTEXITCODE -ne 0) { throw "Session command failed: $Result" }
    return ($Result -join "`n")
}

function Start-Worker {
    if ($Hours -lt 0.01 -or $Hours -gt 24) { throw 'Use 0.01 to 24 hours.' }
    Invoke-Control 'resume' | Out-Null
    # An on-demand Task Scheduler job survives OpenSSH's child-process cleanup.
    # No trigger, startup action, password, or elevated execution is required.
    $WorkerArguments = @('-NoProfile','-ExecutionPolicy','Bypass','-File',
        (Join-Path $ProjectRoot 'learning.ps1'),'-Action','worker','-Session',$Session,'-Hours',
        $Hours.ToString([Globalization.CultureInfo]::InvariantCulture))
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
    $Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $TaskName -Action $TaskAction -Principal $Principal -Settings $Settings -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 2
    if ((Get-ScheduledTask -TaskName $TaskName).State -ne 'Running') {
        $Info = Get-ScheduledTaskInfo -TaskName $TaskName
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
$Label.Text = "CUDA training on this PC. Limit: $Hours active hours.`nPause saves progress and frees the GPU. Closing this panel keeps the session running."
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
            $Status.Text = Invoke-Control 'status'
        } catch { $Status.Text = $_.Exception.Message }
        finally { $Form.UseWaitCursor = $false }
    })
    $Form.Controls.Add($Button)
    $Buttons += $Button
    $i++
}
$Timer = New-Object Windows.Forms.Timer
$Timer.Interval = 5000
$Timer.Add_Tick({
    try {
        $Value = Invoke-Control 'status' | ConvertFrom-Json
        $Status.Text = "State: $($Value.status)`r`nRound: $($Value.round)   Phase: $($Value.phase)`r`n$($Value.detail)`r`nActive hours: $([Math]::Round($Value.active_seconds/3600,2)) / $([Math]::Round($Value.limit_seconds/3600,2))`r`nTraining step: $($Value.training.step) / $($Value.training.max_steps)`r`nAccepted rounds: $($Value.accepted_rounds -join ', ')`r`nSession: $Session"
    } catch { $Status.Text = $_.Exception.Message }
})
$Timer.Start()
try { [void]$Form.ShowDialog() } finally { $Timer.Stop(); $Timer.Dispose(); $Form.Dispose() }
