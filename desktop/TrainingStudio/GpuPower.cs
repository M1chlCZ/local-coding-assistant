using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Security.Principal;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
namespace TrainingStudio;

public sealed record PowerGpu(string Uuid,string Name,double Minimum,double Maximum,double Default,double Limit,double Draw)
{
    public bool Supported => new[]{Minimum,Maximum,Default,Limit}.All(double.IsFinite) && Minimum>0 && Minimum<=Default && Default<=Maximum;
    public int MinimumPercent => Supported ? (int)Math.Ceiling(Minimum/Default*20)*5 : 100;
    public double Watts(int percent)
    {
        if(!Supported || percent<MinimumPercent || percent>100 || percent%5!=0) throw new ArgumentException("Choose a supported power preset between the GPU minimum and 100%.");
        return Math.Round(Default*percent/100,2);
    }
    public override string ToString()=>Name;
}
public sealed record PowerSetting(Profile Profile,string Session,string Uuid,int Percent);

internal static class GpuPower
{
    static string SettingsFile=>Path.Combine(Profile.Home,"gpu-power.json");
    static string Nvidia=>Path.Combine(Environment.SystemDirectory,"nvidia-smi.exe");
    internal static void ValidateUuid(string uuid)
    {
        if(!Regex.IsMatch(uuid,@"\AGPU-[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\z")) throw new ArgumentException("Invalid NVIDIA GPU identifier.");
    }
    public static PowerSetting? Load()=>File.Exists(SettingsFile)?JsonSerializer.Deserialize<PowerSetting>(File.ReadAllText(SettingsFile)):null;
    public static PowerGpu[] Parse(string text)=>text.Split(['\r','\n'],StringSplitOptions.RemoveEmptyEntries).Select(line=>
    {
        var f=line.Split(',').Select(s=>s.Trim().Trim('"')).ToArray();
        if(f.Length!=7) throw new IOException("Unexpected NVIDIA power response.");
        ValidateUuid(f[0]);
        double Number(int i)=>double.TryParse(f[i],NumberStyles.Float,CultureInfo.InvariantCulture,out var n)?n:double.NaN;
        return new PowerGpu(f[0],f[1],Number(2),Number(3),Number(4),Number(5),Number(6));
    }).ToArray();
    public static async Task<PowerGpu[]> Read()
    {
        if(!File.Exists(Nvidia))throw new IOException("NVIDIA power control is unavailable. Install the Windows NVIDIA driver first.");
        var start=new ProcessStartInfo(Nvidia){UseShellExecute=false,CreateNoWindow=true,RedirectStandardOutput=true,RedirectStandardError=true};
        start.ArgumentList.Add("--query-gpu=uuid,name,power.min_limit,power.max_limit,power.default_limit,power.limit,power.draw");
        start.ArgumentList.Add("--format=csv,noheader,nounits");
        return Parse(await Backend.Run(start,timeout:15));
    }
    static string TaskName(string uuid,int percent)=>uuid+"-"+percent.ToString(CultureInfo.InvariantCulture);
    // The installer grants normal users read/run only, never permission to replace the elevated action.
    public static async Task Install(string uuid)
    {
        ValidateUuid(uuid);
        if(!new WindowsPrincipal(WindowsIdentity.GetCurrent()).IsInRole(WindowsBuiltInRole.Administrator))throw new IOException("Administrator permission is needed once to enable GPU power control.");
        var gpu=(await Read()).Single(g=>g.Uuid==uuid);
        if(!gpu.Supported)throw new IOException("This GPU does not expose adjustable power limits.");
        var tasks=Enumerable.Range(0,(100-gpu.MinimumPercent)/5+1).Select(i=>
        {
            int percent=gpu.MinimumPercent+i*5;
            return new{name=TaskName(uuid,percent),watts=gpu.Watts(percent).ToString(CultureInfo.InvariantCulture)};
        }).ToArray();
        string payload=Convert.ToBase64String(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(new{tasks,uuid,nvidia=Nvidia,exe=Path.Combine(Environment.SystemDirectory,"WindowsPowerShell","v1.0","powershell.exe")})));
        await Backend.PowerShell("""
        $ErrorActionPreference='Stop'
        $d=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('PAYLOAD')) | ConvertFrom-Json
        $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
        $sddl='O:BAG:BAD:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;GRGX;;;'+$sid+')'
        $service=New-Object -ComObject Schedule.Service
        $service.Connect()
        try { $folder=$service.GetFolder('\TrainingStudioPower') } catch { $folder=$service.GetFolder('\').CreateFolder('TrainingStudioPower',$sddl) }
        $folder.SetSecurityDescriptor($sddl,0)
        foreach($item in $d.tasks) {
            $task=$service.NewTask(0)
            $task.RegistrationInfo.Description='Training Studio: fixed NVIDIA power preset. No automatic triggers.'
            $task.Principal.UserId=$sid; $task.Principal.LogonType=3; $task.Principal.RunLevel=1
            $task.Settings.Enabled=$true; $task.Settings.AllowDemandStart=$true; $task.Settings.ExecutionTimeLimit='PT30S'
            $task.Settings.DisallowStartIfOnBatteries=$false; $task.Settings.StopIfGoingOnBatteries=$false; $task.Settings.MultipleInstances=2
            $command="& '"+$d.nvidia.Replace("'","''")+"' -i "+$d.uuid+" -pl "+$item.watts+"; exit `$LASTEXITCODE"
            $action=$task.Actions.Create(0); $action.Path=$d.exe
            $action.Arguments='-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
            $folder.RegisterTaskDefinition($item.name,$task,6,$sid,$null,3,$sddl) | Out-Null
        }
        """.Replace("PAYLOAD",payload));
    }
    public static async Task<bool> Enabled(string uuid)
    {
        ValidateUuid(uuid);
        return (await Backend.PowerShell("$ErrorActionPreference='Stop'; $s=New-Object -ComObject Schedule.Service; $s.Connect(); try { $null=$s.GetFolder('\\TrainingStudioPower').GetTask('"+TaskName(uuid,100)+"'); 'yes' } catch { 'no' }")).Trim()=="yes";
    }
    public static async Task Enable(string uuid)
    {
        ValidateUuid(uuid);
        var start=new ProcessStartInfo(Environment.ProcessPath!){UseShellExecute=true,Verb="runas"};
        start.ArgumentList.Add("--enable-gpu-power"); start.ArgumentList.Add(uuid);
        using var process=Process.Start(start)??throw new IOException("Power setup did not start.");
        await process.WaitForExitAsync();
        if(process.ExitCode!=0 || !await Enabled(uuid))throw new IOException("Power setup was cancelled or failed. No training setting was changed.");
    }
    static async Task Apply(string uuid,int percent)
    {
        ValidateUuid(uuid);
        var gpu=(await Read()).Single(g=>g.Uuid==uuid);
        double watts=gpu.Watts(percent);
        if(Math.Abs(gpu.Limit-watts)<0.1)return;
        await Backend.PowerShell("$ErrorActionPreference='Stop'; $s=New-Object -ComObject Schedule.Service; $s.Connect(); $null=$s.GetFolder('\\TrainingStudioPower').GetTask('"+TaskName(uuid,percent)+"').Run($null)");
        for(int i=0;i<20;i++)
        {
            await Task.Delay(250);
            if(Math.Abs((await Read()).Single(g=>g.Uuid==uuid).Limit-watts)<0.1)return;
        }
        throw new IOException("NVIDIA did not confirm the requested power limit. Enable power control again if the driver changed.");
    }
    public static int TargetPercent(int percent,JsonNode state)=>state["status"]?.ToString()=="running" && state["desired"]?.ToString()=="resume" && state["worker_alive"]?.GetValue<bool>()==true ? percent : 100;
    static async Task Locked(Func<Task> action)
    {
        Directory.CreateDirectory(Profile.Home);
        FileStream? lease=null;
        for(int i=0;i<150 && lease is null;i++)
        {
            try { lease=new FileStream(Path.Combine(Profile.Home,"gpu-power.lock"),FileMode.OpenOrCreate,FileAccess.ReadWrite,FileShare.None); }
            catch(IOException) { await Task.Delay(100); }
        }
        if(lease is null)throw new IOException("Another GPU power change is still running. Try again.");
        using(lease)await action();
    }
    public static Task Save(Backend backend,string session,string uuid,int percent)=>Locked(async()=>
    {
        Backend.ValidateSession(session); ValidateUuid(uuid); backend.Profile.Validate();
        var gpu=(await Read()).Single(g=>g.Uuid==uuid); gpu.Watts(percent);
        if(!await Enabled(uuid))throw new IOException("Select Enable power control first.");
        var previous=Load();
        if(previous is not null && previous.Uuid!=uuid)await Apply(previous.Uuid,100);
        var state=await backend.Call(new{action="status",session});
        await Apply(uuid,TargetPercent(percent,state));
        File.WriteAllText(SettingsFile+".tmp",JsonSerializer.Serialize(new PowerSetting(backend.Profile,session,uuid,percent)));
        File.Move(SettingsFile+".tmp",SettingsFile,true);
    });
    public static Task Sync(Backend backend,string session,bool restore=false)=>Locked(async()=>
    {
        var setting=Load();
        if(setting is null || setting.Profile!=backend.Profile || setting.Session!=session)return;
        var state=restore?null:await backend.Call(new{action="status",session});
        await Apply(setting.Uuid,restore?100:TargetPercent(setting.Percent,state!));
    });
}
