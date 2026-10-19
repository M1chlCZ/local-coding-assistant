using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Runtime.InteropServices;
using System.Windows.Interop;
namespace TrainingStudio;
public partial class MainWindow
{
    [DllImport("dwmapi.dll")] static extern int DwmGetWindowAttribute(IntPtr window,int attribute,out int value,int size);
    internal int CheckBackdrop()
    {
        int result=DwmGetWindowAttribute(new WindowInteropHelper(this).Handle,38,out int value,sizeof(int));
        if(Environment.OSVersion.Version.Build>=22621 && result!=0)throw new IOException("Could not inspect native backdrop");
        if(AllowsTransparency)throw new IOException("Layered WPF transparency must stay disabled");
        return value;
    }
    internal void CheckControlStates()
    {
        LearningControls.Check();
        if(QualityDetail.FontWeight!=FontWeights.Normal)throw new IOException("Navigation weight leaked into page content");
        if(((SolidColorBrush)QualityTitle.Foreground).Color!=((SolidColorBrush)Application.Current.Resources["TextBrush"]).Color)
            throw new IOException("Navigation accent leaked into page content");
        var saved=latest;
        try {
            foreach(string status in new[]{"running","paused","stopped","completed","interrupted","failed","pausing","stopping","waiting"}) {
                latest=saved!.DeepClone();latest["status"]=status;latest["desired"]=status is "paused" or "pausing"?"pause":status is "stopped" or "stopping"?"stop":"resume";
                latest["worker_alive"]=status is not("interrupted" or "stopped" or "completed");latest["active_seconds"]=10;
                UpdateControls();UpdateLayout();
                if(PrimaryText.Text!=ControlView().Button || StatusTitle.Text!=(ControlView().Title=="Running"?"Learning from execution":ControlView().Title))throw new IOException("State text differs from controls");
                var position=PrimaryButton.TranslatePoint(new Point(),this);
                if(position.X<0 || position.X+PrimaryButton.ActualWidth>ActualWidth)throw new IOException("Main control is clipped");
            }
            latest=saved!.DeepClone();
            latest["continuous"]=false;
            latest["quality"]!["label"]="Improved on these tests";
            latest["quality"]!["tone"]="good";
            latest["quality"]!["total"]=10;
            latest["report"]=new System.Text.Json.Nodes.JsonObject();
            Render(latest);
            if(QualityTitle.Text!="Improved on these tests" || DecisionTitle.Text!=QualityTitle.Text)
                throw new IOException("Finite experiment verdict was replaced by a promotion decision");
            latest["continuous"]=true;latest["report"]!["accepted"]=false;latest["quality"]!["total"]=0;
            Render(latest);
            if(DecisionTitle.Text!="More verified examples needed" || GainedCount.Text!="—")
                throw new IOException("Collection-only round was shown as a rejected candidate");
            latest["quality"]!["total"]=10;Render(latest);
            if(DecisionTitle.Text!="Previous update rejected")throw new IOException("Rejected candidate was not identified");
            latest["report"]!["accepted"]=true;Render(latest);
            if(DecisionTitle.Text!="Update accepted")throw new IOException("Accepted candidate was not identified");
            latest["continuous"]=false;latest["status"]="completed";latest["phase"]="completed";
            latest["stage_progress"]=null;latest["task_preview"]=null;
            latest["report"]=System.Text.Json.Nodes.JsonNode.Parse("""{"pilot_gate":{},"comparisons":{"training_effect":{"repairs":{"python":{"before":0,"after":0,"total":8}}}}}""");
            Render(latest);
            if(Activity.Text!="Experiment completed" || CurrentTask.Text!="Comparison completed" ||
                ProgressText.Text!="All scheduled checks finished" || StageProgress.Value!=100 || ReplComparison.Text!="0 → 0 / 8" || ChartHint.Text.StartsWith("Waiting"))
                throw new IOException("Completed pilot shows stale progress or omits its measured REPL result");
            latest["report"]!["pilot_gate"]=null;Render(latest);
            if(ReplComparison.Text!="—")throw new IOException("Direct repair tests were mislabeled as REPL results");
        } finally {latest=saved;Render(saved!);UpdateControls();}
    }
    internal async Task CheckRefreshBehavior()
    {
        timer.Stop();
        int changes=0;
        DependencyPropertyChangedEventHandler changed=(_,_)=>changes++;
        SessionPicker.IsEnabledChanged+=changed; Controls.IsEnabledChanged+=changed;
        var release=new TaskCompletionSource();
        Task? poll=null,command=null;
        try
        {
            if(!Controls.IsEnabled || !SessionPicker.IsEnabled)throw new IOException("UI check needs a connected, idle dashboard.");
            poll=Safe(()=>release.Task,false);
            if(!Controls.IsEnabled || !SessionPicker.IsEnabled || changes!=0)throw new IOException("Background refresh disabled the toolbar.");
            bool ran=false;
            command=Safe(()=>{ran=true;return Task.CompletedTask;});
            if(ran)throw new IOException("Command ran concurrently with the status read.");
            release.SetResult();
            await Task.WhenAll(poll,command);
            if(!ran || !Controls.IsEnabled || !SessionPicker.IsEnabled)throw new IOException("Queued command was lost or controls stayed disabled.");
            changes=0;
            await Safe(()=>Task.CompletedTask,false);
            if(changes!=0)throw new IOException("Healthy refresh changed toolbar enabled state.");
        }
        finally
        {
            release.TrySetResult();
            if(poll is not null)await poll;
            if(command is not null)await command;
            SessionPicker.IsEnabledChanged-=changed; Controls.IsEnabledChanged-=changed;
        }
    }
    internal static void CheckThemeContrast()
    {
        foreach(string key in new[]{"CodeKeywordBrush","StepNumberTextBrush","BorderBrush","PositiveBrush","NegativeBrush"})
            if(Appearance.HighContrastColor(key)!=SystemColors.WindowTextColor)throw new IOException("High contrast hides "+key);
        if(Duration(172801)!="2d 00:00:01")throw new IOException("Learning clock wrapped after 24 hours");
        double Light(string key,byte backdrop=0)
        {
            var c=((SolidColorBrush)Application.Current.Resources[key]).Color;
            if(c.A<255) {
                double alpha=c.A/255.0;
                c=Color.FromRgb((byte)(alpha*c.R+(1-alpha)*backdrop),(byte)(alpha*c.G+(1-alpha)*backdrop),(byte)(alpha*c.B+(1-alpha)*backdrop));
            }
            double Channel(byte b){double v=b/255.0;return v<=0.04045?v/12.92:Math.Pow((v+0.055)/1.055,2.4);}
            return 0.2126*Channel(c.R)+0.7152*Channel(c.G)+0.0722*Channel(c.B);
        }
        void Check(string text,string background)
        {
            foreach(byte backdrop in new byte[]{0,255}) {
                double a=Light(text),b=Light(background,backdrop);
                if((Math.Max(a,b)+0.05)/(Math.Min(a,b)+0.05)<4.5)throw new IOException($"Low theme contrast: {text} on {background}");
            }
        }
        foreach(var background in new[]{"WindowBrush","BackdropTintBrush","CardBrush","NoticeBrush","InsetBrush","StepBrush","QualityGoodBrush","QualityBadBrush","QualityMixedBrush","QualityNeutralBrush"})
        {
            Check("TextBrush",background);Check("MutedBrush",background);
        }
        Check("CodeKeywordBrush","InsetBrush");
        Check("StepNumberTextBrush","StepNumberBrush");
        Check("LogTextBrush","LogBrush");
        Check("PrimaryTextBrush","PrimaryBrush");
    }
}
