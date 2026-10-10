using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace TrainingStudio;

public sealed record Profile(string Distribution, string User, string Root, string Python)
{
    public static string Home => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "TrainingStudio");
    public static string FileName => Path.Combine(Home, "connection.json");
    public static Profile Load() => File.Exists(FileName) ? JsonSerializer.Deserialize<Profile>(File.ReadAllText(FileName))! : new("", "", "", ".cache/rlm-env/bin/python");
    public void Validate()
    {
        if (new[]{Distribution, User, Root, Python}.Any(s => string.IsNullOrWhiteSpace(s) || s.Any(char.IsControl) || s.Contains('"')) || !Root.StartsWith('/') || Root.Contains("/../") || Distribution.StartsWith('-') || User.StartsWith('-') || Python.StartsWith('-'))
            throw new ArgumentException("Enter a WSL distribution, Linux user, absolute Linux project folder and Python path.");
    }
    public void Save() { Validate(); Directory.CreateDirectory(Home); File.WriteAllText(FileName, JsonSerializer.Serialize(this)); }
}

public sealed record StartupOptions(bool ResumeLearningOnLogin=true, bool OpenStudioOnLogin=false)
{
    static string FileName => Path.Combine(Profile.Home,"startup.json");
    public static StartupOptions Load() => File.Exists(FileName) ? JsonSerializer.Deserialize<StartupOptions>(File.ReadAllText(FileName))! : new();
    public void Save()
    {
        Directory.CreateDirectory(Profile.Home);
        File.WriteAllText(FileName+".tmp",JsonSerializer.Serialize(this));
        File.Move(FileName+".tmp",FileName,true);
    }
}

public sealed class Backend(Profile profile)
{
    public Profile Profile { get; } = profile;
    public static void ValidateSession(string name)
    {
        if (!Regex.IsMatch(name, @"\A[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}\z")) throw new ArgumentException("Invalid experiment name.");
    }
    ProcessStartInfo Wsl(params string[] command)
    {
        Profile.Validate();
        var start = new ProcessStartInfo("wsl.exe") { UseShellExecute=false, CreateNoWindow=true, RedirectStandardInput=true, RedirectStandardOutput=true, RedirectStandardError=true, StandardOutputEncoding=Encoding.UTF8, StandardErrorEncoding=Encoding.UTF8 };
        foreach (var arg in new[]{"-d",Profile.Distribution,"-u",Profile.User,"--cd",Profile.Root,"--exec"}.Concat(command)) start.ArgumentList.Add(arg);
        start.Environment["WSL_UTF8"]="1";
        return start;
    }
    public static async Task<string> Run(ProcessStartInfo start, string? input=null, int timeout=45)
    {
        using var process=Process.Start(start) ?? throw new IOException("Could not start " + start.FileName);
        var output=process.StandardOutput.ReadToEndAsync(); var error=process.StandardError.ReadToEndAsync();
        if (start.RedirectStandardInput) { if(input is not null) await process.StandardInput.WriteAsync(input); process.StandardInput.Close(); }
        try { await process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(timeout)); }
        catch (TimeoutException) { process.Kill(true); throw new IOException("The connection timed out. Saved training was not stopped. Check WSL and reconnect."); }
        var text=await output; var stderr=await error;
        if(process.ExitCode!=0) throw new IOException(ReadableError(string.IsNullOrWhiteSpace(text) ? stderr : text));
        return text;
    }
    internal static string ReadableError(string text)
    {
        try { if(JsonNode.Parse(text)?["error"] is { } error) return error.ToString(); } catch(JsonException) { }
        int xml=text.IndexOf("<Objs",StringComparison.Ordinal);
        if(xml>=0)
        {
            try
            {
                var doc=System.Xml.Linq.XDocument.Parse(text[xml..]);
                var lines=doc.Descendants().Where(n=>n.Name.LocalName=="S" && (string?)n.Attribute("S")=="Error").Select(n=>System.Xml.XmlConvert.DecodeName(n.Value));
                var message=string.Join(Environment.NewLine,lines);
                if(!string.IsNullOrWhiteSpace(message)) return message.Trim();
            }
            catch(System.Xml.XmlException) { }
        }
        return text.Trim();
    }
    public async Task<JsonNode> Call(object request, int timeout=45)
    {
        var text=await Run(Wsl(Profile.Python,"desktop_bridge.py"),JsonSerializer.Serialize(request),timeout);
        var value=JsonNode.Parse(text) ?? throw new IOException("Empty worker response");
        if(value["error"] is not null) throw new IOException(value["error"]!.ToString());
        return value;
    }
    public static async Task<string[]> Distributions()
    {
        var start=new ProcessStartInfo("wsl.exe") { UseShellExecute=false, CreateNoWindow=true, RedirectStandardOutput=true, RedirectStandardError=true, StandardOutputEncoding=Encoding.Unicode };
        start.ArgumentList.Add("--list"); start.ArgumentList.Add("--quiet");
        return (await Run(start)).Replace("\0", "").Split(['\r','\n'],StringSplitOptions.RemoveEmptyEntries|StringSplitOptions.TrimEntries);
    }
    public async Task EnsureWorker(string session)
    {
        ValidateSession(session); Profile.Validate();
        var folder=Path.Combine(Profile.Home,"workers"); Directory.CreateDirectory(folder);
        var id=Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(Profile)+session)))[..16];
        var config=Path.Combine(folder,id+".json"); File.WriteAllText(config,JsonSerializer.Serialize(Profile));
        var exe=Environment.ProcessPath ?? throw new IOException("Cannot locate installed app");
        if(Path.GetFileNameWithoutExtension(exe).Equals("dotnet",StringComparison.OrdinalIgnoreCase)) throw new IOException("Run the published TrainingStudio.exe to install the background worker.");
        var payload=Convert.ToBase64String(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(new{exe,config,session,name="TrainingStudio-"+id,automatic=StartupOptions.Load().ResumeLearningOnLogin})));
        // Data crosses PowerShell as base64 JSON, never interpolated executable syntax.
        var script="""
        $ErrorActionPreference='Stop'
        $d=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('PAYLOAD')) | ConvertFrom-Json
        $owner=[Security.Principal.WindowsIdentity]::GetCurrent().Name
        $arguments='--worker "' + $d.config + '" ' + $d.session
        $action=New-ScheduledTaskAction -Execute $d.exe -Argument $arguments
        $triggers=@()
        $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
        if($d.automatic) {
            $triggers=@((New-ScheduledTaskTrigger -AtLogOn -User $owner),(New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) -RepetitionInterval ([TimeSpan]::FromMinutes(5))))
            $settings.RestartCount=3; $settings.RestartInterval='PT1M'
        }
        $principal=New-ScheduledTaskPrincipal -UserId $owner -LogonType Interactive -RunLevel Limited
        $task=New-ScheduledTask -Action $action -Settings $settings -Principal $principal
        $task.Triggers=$triggers
        Register-ScheduledTask -TaskName $d.name -InputObject $task -Force | Out-Null
        Start-ScheduledTask -TaskName $d.name
        """.Replace("PAYLOAD",payload);
        await PowerShell(script);
    }
    internal static async Task<string> PowerShell(string script)
    {
        var start=new ProcessStartInfo("powershell.exe"){UseShellExecute=false,CreateNoWindow=true,RedirectStandardOutput=true,RedirectStandardError=true,StandardOutputEncoding=Encoding.UTF8,StandardErrorEncoding=Encoding.UTF8};
        foreach(var arg in new[]{"-NoProfile","-NonInteractive","-EncodedCommand",Convert.ToBase64String(Encoding.Unicode.GetBytes("$ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new()\n"+script))}) start.ArgumentList.Add(arg);
        return await Run(start);
    }
    public static async Task ConfigureStartup(StartupOptions options)
    {
        var exe=Environment.ProcessPath ?? throw new IOException("Cannot locate installed app");
        var payload=Convert.ToBase64String(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(new{exe,automatic=options.ResumeLearningOnLogin,open=options.OpenStudioOnLogin})));
        await PowerShell("""
        $ErrorActionPreference='Stop'
        $d=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('PAYLOAD')) | ConvertFrom-Json
        $owner=[Security.Principal.WindowsIdentity]::GetCurrent().Name
        Get-ScheduledTask -TaskName 'TrainingStudio-*' -ErrorAction SilentlyContinue | ForEach-Object {
            $task=$_
            if($task.Actions.Arguments -notmatch '^--worker ') { return }
            $task.Triggers=@()
            $task.Settings.RestartCount=0; $task.Settings.RestartInterval=$null
            if($d.automatic) {
                $task.Triggers=@((New-ScheduledTaskTrigger -AtLogOn -User $owner),(New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) -RepetitionInterval ([TimeSpan]::FromMinutes(5))))
                $task.Settings.RestartCount=3; $task.Settings.RestartInterval='PT1M'
            }
            Set-ScheduledTask -InputObject $task | Out-Null
        }
        if($d.open) {
            $action=New-ScheduledTaskAction -Execute $d.exe
            $trigger=New-ScheduledTaskTrigger -AtLogOn -User $owner
            $principal=New-ScheduledTaskPrincipal -UserId $owner -LogonType Interactive -RunLevel Limited
            $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
            Register-ScheduledTask -TaskName 'TrainingStudioUI' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
        } else {
            Get-ScheduledTask -TaskName 'TrainingStudioUI' -ErrorAction SilentlyContinue | Unregister-ScheduledTask -Confirm:$false
        }
        exit 0
        """.Replace("PAYLOAD",payload));
        options.Save();
    }
    [DllImport("kernel32.dll")] static extern uint SetThreadExecutionState(uint flags);
    public static async Task Worker(string config, string session)
    {
        ValidateSession(session);
        var profile=JsonSerializer.Deserialize<Profile>(File.ReadAllText(config)) ?? throw new IOException("Invalid connection profile");
        var backend=new Backend(profile);
        var state=await backend.Call(new{action="status",session});
        if(new[]{"completed","stopped","budget_exhausted"}.Contains(state["status"]?.ToString()) || state["worker_alive"]?.GetValue<bool>()==true) return;
        Directory.CreateDirectory(Profile.Home);
        var log=Path.Combine(Profile.Home,"workers",Path.GetFileNameWithoutExtension(config)+".log");
        using var process=Process.Start(backend.Wsl(profile.Python,"focused_experiment.py","run","--session",".cache/learning/"+session)) ?? throw new IOException("Could not start WSL worker");
        process.StandardInput.Close();
        async Task Drain(StreamReader stream)
        {
            while(await stream.ReadLineAsync() is { } line) { lock(LogGate) File.AppendAllText(log,line+Environment.NewLine); }
        }
        var stdout=Drain(process.StandardOutput); var stderr=Drain(process.StandardError);
        try
        {
            while(!process.HasExited)
            {
                try
                {
                    state=await backend.Call(new{action="status",session});
                    bool active=new[]{"pausing","stopping"}.Contains(state["status"]?.ToString()) || (new[]{"running","waiting"}.Contains(state["status"]?.ToString()) && state["desired"]?.ToString()=="resume");
                    SetThreadExecutionState(active ? 0x80000001 : 0x80000000);
                }
                catch(Exception ex) { lock(LogGate) File.AppendAllText(log,ex.Message+Environment.NewLine); }
                await Task.WhenAny(process.WaitForExitAsync(), Task.Delay(10000));
            }
            await Task.WhenAll(stdout,stderr);
            if(process.ExitCode!=0) throw new IOException("WSL worker exited " + process.ExitCode + ". See " + log);
        }
        finally { SetThreadExecutionState(0x80000000); }
    }
    static readonly object LogGate=new();
}
