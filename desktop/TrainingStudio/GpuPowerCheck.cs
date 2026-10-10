using System.IO;
using System.Text.Json.Nodes;
namespace TrainingStudio;
internal static class GpuPowerCheck
{
    public static void Validate()
    {
        var gpu=GpuPower.Parse("GPU-11111111-2222-3333-4444-555555555555, Example GPU, 250.00, 300.00, 300.00, 300.00, 280.00\n").Single();
        if(gpu.MinimumPercent!=85 || gpu.Watts(95)!=285 || gpu.Watts(90)!=270 || gpu.Watts(85)!=255) throw new IOException("Power percentage conversion failed.");
        foreach(var value in new[]{80,101,105,0,-5,91})
        {
            bool rejected=false;
            try { gpu.Watts(value); } catch(ArgumentException) { rejected=true; }
            if(!rejected) throw new IOException("Unsupported power percentage accepted.");
        }
        if(GpuPower.Parse("GPU-11111111-2222-3333-4444-555555555555, Unsupported, N/A, N/A, N/A, N/A, N/A\n").Single().Supported) throw new IOException("Unsupported GPU enabled.");
        foreach(var status in new[]{"paused","pausing","stopped","stopping","completed","failed","budget_exhausted","interrupted","waiting"})
            if(GpuPower.TargetPercent(90,JsonNode.Parse($"{{\"status\":\"{status}\",\"desired\":\"resume\",\"worker_alive\":true}}")!)!=100) throw new IOException("Inactive GPU was not restored.");
        var running=JsonNode.Parse("{\"status\":\"running\",\"desired\":\"resume\",\"worker_alive\":true}")!;
        if(GpuPower.TargetPercent(90,running)!=90)throw new IOException("Active training cap was not applied.");
        running["desired"]="pause";
        if(GpuPower.TargetPercent(90,running)!=100)throw new IOException("Pause request did not immediately restore power.");
        running["desired"]="resume"; running["worker_alive"]=false;
        if(GpuPower.TargetPercent(90,running)!=100)throw new IOException("Missing worker retained power cap.");
    }
    public static async Task Hardware(string session,string output)
    {
        var backend=new Backend(Profile.Load());
        var state=await backend.Call(new{action="status",session});
        if(GpuPower.TargetPercent(95,state)!=95)throw new IOException("Hardware check needs an already running experiment.");
        var gpu=(await GpuPower.Read()).First(g=>g.Supported);
        var original=GpuPower.Load();
        try
        {
            foreach(int percent in new[]{95,90,85,100}.Where(p=>p>=gpu.MinimumPercent))
            {
                await GpuPower.Save(backend,session,gpu.Uuid,percent);
                var actual=(await GpuPower.Read()).Single(g=>g.Uuid==gpu.Uuid);
                if(Math.Abs(actual.Limit-gpu.Watts(percent))>0.1)throw new IOException("Hardware limit mismatch.");
            }
        }
        finally
        {
            if(original is not null)await GpuPower.Save(new Backend(original.Profile),original.Session,original.Uuid,original.Percent);
            else await GpuPower.Save(backend,session,gpu.Uuid,100);
        }
        File.WriteAllText(output,"Hardware presets passed; original setting restored (100% when unset).");
    }

}
