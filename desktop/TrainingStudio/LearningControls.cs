namespace TrainingStudio;

internal sealed record LearningControls(string Title,string Hint,string Icon,string Button,string Action,bool CanPause,bool Enabled,string Tone)
{
    public static LearningControls For(string status,string desired,bool alive,bool started,bool continuous=false)
    {
        if(status=="empty")return new("No session yet","Create a session to prepare learning. Nothing starts automatically.","\uE710","Create session","new",false,true,"Neutral");
        if(status=="unknown")return new("Status unavailable","Check the connection before changing learning.","\uE946","Start","none",false,false,"Neutral");
        if(status=="stopped" && continuous)return new("Stopped","Resume continues your saved progress. Learning stays off after login until you resume.","\uE71A","Resume","resume",false,true,"Neutral");
        if(status=="stopped")return new("Stopped","This session has ended. Its saved results are still available.","\uE71A","Start new session","new",false,true,"Neutral");
        if(status is "completed" or "budget_exhausted")return new("Finished","Saved results are ready. Start a new session to keep experimenting.","\uE73E","Start new session","new",false,true,"Neutral");
        if(status=="stopping" || desired=="stop")return new("Stopping","Saving progress and releasing your GPU…","\uE71A","Stopping…","none",false,false,"Neutral");
        if(status=="pausing" || (desired=="pause" && status!="paused"))return new("Pausing","Saving progress. Wait for Paused before gaming.","\uE769","Pausing…","none",false,false,"Neutral");
        if(status=="paused")return new(started?"Paused":"Ready",started?"This session has released the GPU. Start continues saved progress.":"Start learning whenever you are ready.",started?"\uE769":"\uE768","Start","resume",false,true,"Neutral");
        if(status=="failed")return new("Needs attention","Open Activity for the error. Try again after fixing it.","\uE7BA","Try again","resume",false,true,"Mixed");
        if(!alive || status=="interrupted")return new("Not running","The worker is offline. Start recovers saved progress.","\uE768","Start","resume",false,true,"Neutral");
        if(status=="waiting")return new("Waiting to retry","A temporary problem delayed learning. Studio will retry.","\uE823","Stop","stop",true,true,"Mixed");
        return new("Running","Learning continues in the background, even when you close Studio.","\uE768","Stop","stop",true,true,"Good");
    }
    public static void Check()
    {
        var empty=For("empty","",false,false);
        if(empty.Action!="new" || !empty.Enabled || empty.CanPause || empty.Button!="Create session")
            throw new InvalidOperationException("An empty connected backend must offer session creation, not resume");
        var running=For("running","resume",true,true);
        if(running.Action!="stop" || !running.CanPause)throw new InvalidOperationException("Running controls are incorrect");
        foreach(var status in new[]{"paused","interrupted","failed"})
            if(For(status,status=="paused"?"pause":"resume",status=="paused",true).Action!="resume")throw new InvalidOperationException("Recovery action is incorrect");
        foreach(var status in new[]{"stopped","completed","budget_exhausted"})
            if(For(status,"stop",false,true).Action!="new")throw new InvalidOperationException("Ended session must stay ended");
        if(For("stopped","stop",false,true,true).Action!="resume")throw new InvalidOperationException("Stopped loop must allow explicit Resume");
        foreach(var status in new[]{"completed","budget_exhausted"})
            if(For(status,"resume",false,true,true).Action!="new")throw new InvalidOperationException("Finished loop cannot restart");
        if(For("running","pause",true,true).Enabled || For("running","stop",true,true).Enabled || For("unknown","",false,false).Enabled)
            throw new InvalidOperationException("Pending or disconnected controls must be disabled");
        if(For("running","resume",false,true).Title=="Running")throw new InvalidOperationException("An absent worker cannot appear running");
    }
}
