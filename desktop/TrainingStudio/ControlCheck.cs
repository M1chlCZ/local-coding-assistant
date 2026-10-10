using System.IO;
using System.Text.Json.Nodes;
namespace TrainingStudio;

// A separate non-GPU fixture exercises the real Windows scheduler and WSL controls.
internal static class ControlCheck
{
    public static void ValidateInputs()
    {
        if(Backend.ReadableError("{\"error\":\"Readable error\"}")!="Readable error") throw new IOException("JSON error parser failed");
        if(Backend.ReadableError("#< CLIXML\n<Objs><S S=\"Error\">Failed_x000D__x000A_</S></Objs>")!="Failed") throw new IOException("PowerShell error parser failed");
        foreach(var invalid in new[]{"../escape","a/b","","x\n"})
        {
            bool rejected=false;
            try { Backend.ValidateSession(invalid); } catch(ArgumentException) { rejected=true; }
            if(!rejected) throw new IOException("Invalid session accepted");
        }
    }
    public static async Task Run(string session,string output)
    {
        if(!session.StartsWith("desktop-smoke-",StringComparison.Ordinal)) throw new ArgumentException("Use a dedicated desktop-smoke- fixture.");
        ValidateInputs();
        var backend=new Backend(Profile.Load());
        async Task<JsonNode> Read()=>await backend.Call(new{action="status",session});
        async Task<JsonNode> Wait(Func<JsonNode,bool> match)
        {
            for(int i=0;i<60;i++){var s=await Read();if(match(s))return s;await Task.Delay(500);}
            throw new IOException("Control state did not arrive within 30 seconds.");
        }
        await backend.Call(new{action="resume",session});
        await backend.EnsureWorker(session);
        await Wait(s=>s["status"]?.ToString()=="running" && s["worker_alive"]?.GetValue<bool>()==true);
        await Task.Delay(3000);
        await backend.Call(new{action="pause",session});
        var paused=await Wait(s=>s["status"]?.ToString()=="paused");
        double clock=paused["active_seconds"]!.GetValue<double>();
        await Task.Delay(2500);
        if((await Read())["active_seconds"]!.GetValue<double>()!=clock)throw new IOException("Paused clock changed.");
        await backend.Call(new{action="resume",session});
        await Wait(s=>s["active_seconds"]!.GetValue<double>()>clock);
        await backend.Call(new{action="stop",session});
        await Wait(s=>s["status"]?.ToString()=="stopped");
        await backend.EnsureWorker(session);
        await Task.Delay(2000);
        if((await Read())["status"]?.ToString()!="stopped")throw new IOException("Recovery resumed a stopped experiment.");
        File.WriteAllText(output,"{\"scheduler\":true,\"pause_clock_preserved\":true,\"resume\":true,\"stop_respected_on_recovery\":true}");
    }
}
