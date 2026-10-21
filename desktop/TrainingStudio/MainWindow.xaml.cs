using Microsoft.Win32;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
using System.Windows.Documents;
using System.Text.RegularExpressions;
namespace TrainingStudio;

public partial class MainWindow : Window
{
    Backend? backend;
    JsonNode? latest;
    bool busy, loading, powerBusy, powerLoading;
    readonly SemaphoreSlim operationGate=new(1,1);
    readonly DispatcherTimer timer = new() { Interval=TimeSpan.FromSeconds(5) };
    public TaskCompletionSource<bool> Ready { get; } = new();
    string displayedCode="";
    string? Session => SessionPicker.SelectedItem as string;
    public MainWindow()
    {
        InitializeComponent(); Controls.IsEnabled=false;
        SourceInitialized += (_,_)=>Appearance.TitleBar(this);
        ThemePicker.SelectedIndex=Array.IndexOf(new[]{"System","Light","Dark"},Appearance.Mode);
        try { var p=Profile.Load(); Distro.Text=p.Distribution; LinuxUser.Text=p.User; ProjectRoot.Text=p.Root; PythonPath.Text=p.Python; }
        catch(Exception ex) { Notice.Text="Saved connection could not be read: "+ex.Message; }
        try { var options=StartupOptions.Load(); AutoLearning.IsChecked=options.ResumeLearningOnLogin; AutoStudio.IsChecked=options.OpenStudioOnLogin; StartupSummary.Text=StartupDescription(options); }
        catch(Exception ex) { Notice.Text="Startup preferences could not be read: "+ex.Message; }
        Loaded += async (_,_) => { await Safe(async()=>{if(!string.IsNullOrWhiteSpace(ProjectRoot.Text)) await Connect(); else OpenSettings(3);}); await RefreshPower(true); Ready.TrySetResult(latest is not null); timer.Start(); };
        timer.Tick += async (_,_) => {if(!busy && backend is not null && Session is not null) await Safe(Refresh,false); if(PowerTab.IsSelected && !powerBusy) await RefreshPower(false);};
        Closed += (_,_)=>timer.Stop();
    }
    async Task Safe(Func<Task> work, bool announce=true)
    {
        if(announce)
        {
            if(busy)return;
            busy=true; UpdateControls(); SessionPicker.IsEnabled=false;
            await operationGate.WaitAsync();
        }
        else if(busy || !await operationGate.WaitAsync(0))return;
        try { await work(); if(announce) Notice.Text="Your data and model files stay on this PC."; }
        catch(Exception ex) { latest=null; Notice.Text=ex.Message; ConnectionBadge.Text="Action failed · check details below"; if(!announce) {Activity.Text="Connection unavailable"; ActivityDetail.Text="Displayed results are from the last successful read. Training may still be running."; StageProgress.IsIndeterminate=false;} }
        finally
        {
            if(announce){busy=false;SessionPicker.IsEnabled=true;}
            operationGate.Release(); UpdateControls();
        }
    }
    void UpdateControls()
    {
        Controls.IsEnabled=!busy && backend is not null && latest is not null;
        var view=ControlView();
        PrimaryButton.IsEnabled=view.Enabled;
        PrimaryText.Text=view.Button;
        PrimaryIcon.Text=view.Action=="stop"?"\uE71A":"\uE768";
        System.Windows.Automation.AutomationProperties.SetName(PrimaryButton,view.Button+" learning");
        PrimaryButton.ToolTip=view.Action=="stop"?(latest?["continuous"]?.GetValue<bool>()==true?"Stop learning and automatic restart. Resume continues saved progress.":"End this session. Use Pause if you want to continue it later."):view.Hint;
        PauseButton.Visibility=view.CanPause?Visibility.Visible:Visibility.Collapsed;
        PauseButton.IsEnabled=view.CanPause;
        StatusTitle.Text=view.Title=="Running"?"Learning from execution":view.Title; StatusHint.Text=view.Hint; StatusIcon.Text=view.Icon;
        StatusTile.SetResourceReference(Border.BackgroundProperty,view.Title=="Running"?"PrimaryBrush":"MutedBrush");
    }
    LearningControls ControlView()=>LearningControls.For(S(latest,"status","unknown"),S(latest,"desired",""),latest?["worker_alive"]?.GetValue<bool>()==true,N(latest,"active_seconds")>0,latest?["continuous"]?.GetValue<bool>()==true);
    void OpenSettings(int page){Tabs.SelectedIndex=3;SettingsTabs.SelectedIndex=page;}
    async Task Connect()
    {
        var p=new Profile(Distro.Text.Trim(),LinuxUser.Text.Trim(),ProjectRoot.Text.Trim(),PythonPath.Text.Trim());
        p.Validate(); var next=new Backend(p);
        var data=await next.Call(new{action="list"}); p.Save(); backend=next;
        await LoadSessions(data);
        ConnectionBadge.Text="Connected · local WSL";
    }
    async Task LoadSessions(JsonNode? data=null, string? select=null)
    {
        data ??= await backend!.Call(new{action="list"});
        var names=data["sessions"]!.AsArray().Select(n=>n!["name"]!.ToString()).ToList();
        string? previous=select??Session;
        loading=true; SessionPicker.ItemsSource=names; SessionPicker.SelectedItem=names.Contains(previous??"")?previous:names.FirstOrDefault(); loading=false;
        latest=null;
        RenderHistory(data);
        if(Session is not null) await Refresh();
        else {latest=new JsonObject{["status"]="empty"}; Activity.Text="No experiments yet"; ActivityDetail.Text="Create a learning session in Settings. Nothing is running."; OpenSettings(0);}
    }
    void RenderHistory(JsonNode data)
    {
        var history=(data["history"]?.AsArray() ?? new JsonArray()).Select(n=>new { Experiment=S(n,"session"), Verdict=S(n,"label"), Before=N(n,"before"), After=N(n,"after"), Total=N(n,"total"), Lost=N(n,"lost"), Delta=100*(N(n,"after")-N(n,"before"))/Math.Max(1,N(n,"total")) }).ToList();
        HistoryGrid.ItemsSource=history;
        HistoryChart.Children.Clear();
        if(history.Count==0) HistoryChart.Children.Add(new TextBlock{Text="No completed comparisons yet.",Margin=new Thickness(0,0,0,16)});
        foreach(var item in history)
        {
            var row=new StackPanel{Margin=new Thickness(0,0,0,16)};
            row.Children.Add(new TextBlock{Text=$"{item.Experiment}    {item.Delta:+0.0;-0.0;0.0} percentage points · {item.Lost:0} lost",Margin=new Thickness(0,0,0,5)});
            var bar=new ProgressBar{Minimum=0,Maximum=100,Value=Math.Abs(item.Delta),Height=10};
            bar.SetResourceReference(ProgressBar.ForegroundProperty,item.Delta<0?"NegativeBrush":"PositiveBrush");
            row.Children.Add(bar);
            HistoryChart.Children.Add(row);
        }
    }
    async Task Refresh()
    {
        if(Session is null || backend is null) return;
        string reportKey=S(latest,"last_report_sha256",latest?["report"]?.ToJsonString()??"");
        string session=Session; var currentBackend=backend;
        var result=await currentBackend.Call(new{action="status",session});
        if(session!=Session || currentBackend!=backend)return;
        latest=result; Render(latest);
        if(latest["report"] is not null && reportKey!=S(latest,"last_report_sha256",latest["report"]?.ToJsonString()??"")) RenderHistory(await backend.Call(new{action="list"}));
        ConnectionBadge.Text="Updated "+DateTime.Now.ToString("HH:mm:ss")+" · local WSL";
    }
    static string S(JsonNode? n,string key,string fallback="—")=>n?[key]?.ToString()??fallback;
    static double N(JsonNode? n,string key)=>double.TryParse(n?[key]?.ToString(),NumberStyles.Float,CultureInfo.InvariantCulture,out var v)?v:0;
    internal static string Duration(double seconds)
    {
        var time=TimeSpan.FromSeconds(Math.Max(0,seconds));
        return (time.Days>0?$"{time.Days}d ":"")+time.ToString(@"hh\:mm\:ss");
    }
    void Render(JsonNode state)
    {
        var q=state["quality"]!;
        bool continuous=state["continuous"]?.GetValue<bool>()==true;
        bool finished=S(state,"status")=="completed";
        QualityTitle.Text=continuous && N(state,"accepted_updates")==0?"No update accepted yet":S(q,"label");
        QualityCard.SetResourceReference(Border.BackgroundProperty,"CardBrush");
        QualityDetail.Text=N(q,"total")>0?$"Latest comparison: {N(q,"before"):0} → {N(q,"after"):0} / {N(q,"total"):0} passed. {N(q,"gained"):0} gained · {N(q,"lost"):0} lost.":"Candidates must pass matched checks.";
        string phase=S(state["stage_progress"],"phase",S(state,"phase")).Split('/').Last();
        string stage=S(state,"phase").Split('/').Last();
        if(phase=="evaluate")phase=stage;
        Activity.Text=finished?"Experiment completed":phase switch {
            "prepare" or "validate-source"=>"Checking new examples",
            "verify-teacher"=>"Checking published teacher examples",
            "load-model" or "load_model" or "load_runtime"=>"Loading the model",
            "collect"=>"Solving coding examples",
            "verify-repl"=>"Executing REPL repair lessons",
            "export" or "verify-replay" or "tokenize"=>"Preparing verified examples",
            "train"=>"Training a small update",
            "baseline" or "audit-parent"=>"Testing the current model",
            "candidate" or "audit-candidate"=>"Testing the new candidate",
            "repl-baseline"=>"Checking current REPL skills",
            "repl-candidate"=>"Checking candidate REPL skills",
            "gate" or "finish"=>"Comparing the results",
            "saved"=>"Progress saved",_=>"Working on this learning cycle"};
        ActivityDetail.Text=finished?"Scheduled checks are complete. REPL counts are verified training rows.":"Live counts appear as tasks finish. REPL counts are verified training rows.";
        var p=state["stage_progress"];
        CurrentTask.Text=finished?"Comparison completed":S(p,"task","Preparing the next task");
        CurrentTask.ToolTip=CurrentTask.Text;
        var preview=state["task_preview"];
        CurrentTask.Text=S(preview,"filename",CurrentTask.Text);
        TaskLanguage.Text=S(preview,"language",S(p,"language",""));
        displayedCode=S(preview,"code","");
        CopyCode.IsEnabled=displayedCode.Length>0 && preview?["truncated"]?.GetValue<bool>()!=true;
        CodeLines.Text=string.Join("\n",Enumerable.Range(1,Math.Min(8,displayedCode.Split('\n').Length)));
        CodePreview.Inlines.Clear();
        string snippet=string.Join("\n",displayedCode.Replace("\t","    ").Split('\n').Take(8));
        if(snippet.Length==0){CodeLines.Text="";snippet=finished?"Experiment finished. Results and checkpoints remain saved.":"Source preview appears when a coding task starts.";}
        foreach(string part in Regex.Split(snippet,@"(\b(?:def|return|import|from|if|else|for|while|class|func|package|fn|let|const|var|int|void|print|pub|use|std)\b)")) {
            var run=new Run(part);
            if(Regex.IsMatch(part,@"^[a-z]+$"))run.SetResourceReference(TextElement.ForegroundProperty,"CodeKeywordBrush");
            CodePreview.Inlines.Add(run);
        }
        bool observed=N(state,"collected_examples")>0;
        FirstPasses.Text=observed?$"{N(state,"first_passes"):0} / {N(state,"collected_examples"):0}":"—";
        RepairedCount.Text=observed?$"{N(state,"repaired_examples"):0}":"—";
        ReplCount.Text=N(state,"training_examples")>0?$"{N(state,"repl_examples"):0}":"—";
        AcceptedCount.Text=$"{N(state,"accepted_updates"):0} accepted updates";
        int current=finished?4:stage switch {"prepare"=>0,"collect" or "export"=>1,"train" or "train-agent"=>2,"finish"=>4,_=>3};
        for(int i=0;i<Pipeline.Children.Count;i++) {
            var step=(Border)Pipeline.Children[i];
            step.SetResourceReference(Border.BackgroundProperty,i==current?"StepBrush":"ClearBrush");
            var number=(Border)((Grid)step.Child).Children[0];
            number.SetResourceReference(Border.BackgroundProperty,i==current?"PrimaryBrush":"StepNumberBrush");
            ((TextBlock)number.Child).SetResourceReference(TextBlock.ForegroundProperty,i==current?"PrimaryTextBrush":"StepNumberTextBrush");
            step.SetResourceReference(TextElement.ForegroundProperty,i==current?"PrimaryBrush":"TextBrush");
        }
        double done=N(p,"completed"),total=N(p,"total");
        if(total<=0){done=N(p,"step");total=N(p,"max_steps");}
        StageProgress.IsIndeterminate=total<=0 && S(state,"status")=="running";
        StageProgress.Value=finished?100:total>0?100*done/total:0;
        string units=phase is "train" or "saved"?"training steps":phase.Contains("audit") || phase.Contains("repl-") || phase is "baseline" or "candidate"?"tests":"examples";
        ProgressText.Text=finished?"All scheduled checks finished":total>0?$"{done:0} of {total:0} {units}":"Preparing this step…";
        if(p?["language"] is not null)ProgressText.Text+=" · "+CultureInfo.InvariantCulture.TextInfo.ToTitleCase(S(p,"language"));
        if(S(state,"status")=="running" && N(p,"heartbeat_at")>0 && DateTimeOffset.UtcNow.ToUnixTimeSeconds()-N(p,"heartbeat_at")>120)
            ProgressText.Text+=" · This step is taking longer. Details are in Activity.";
        StageProgress.ToolTip=ProgressText.Text;
        Teaching.Text=continuous
            ?(S(state,"teaching_method") switch {"execution_repair"=>"Execution feedback + REPL repair lessons","teacher_mix"=>"Teacher examples + verified replay",_=>"Verified self-training"})+
                (N(state,"training_examples")>0?$" · {N(state,"training_examples"):0} training rows · {N(state,"correction_examples"):0} corrections":"")
            :"Supervised training experiment";
        Clocks.Text="Training "+Duration(N(state,"training_seconds"))+(continuous?" · No time cutoff":" / "+Duration(N(state,"limit_seconds")))+" · All active work "+Duration(N(state,"active_seconds"));
        if(continuous) {
            ProgressText.Text="Cycle "+N(state,"cycle").ToString("0");

            if(state["report"]?["accepted"]?.GetValue<bool>()==false) {
                QualityTitle.Text="Retained model unchanged";
                QualityCard.SetResourceReference(Border.BackgroundProperty,"CardBrush");
            }
        }
        var report=state["report"];
        var reasons=report?["reasons"]?.AsArray();
        bool measured=N(q,"total")>0;
        DecisionTitle.Text=report is null?"Waiting for a comparison":!continuous?S(q,"label"):
            !measured?"More verified examples needed":report["accepted"]?.GetValue<bool>()==true?"Update accepted":"Previous update rejected";
        string decisionBrush=!measured?"MutedBrush":continuous?(report?["accepted"]?.GetValue<bool>()==true?"PositiveBrush":"NegativeBrush"):
            S(q,"tone") switch {"good"=>"PositiveBrush","bad"=>"NegativeBrush",_=>"MutedBrush"};
        DecisionDot.SetResourceReference(System.Windows.Shapes.Shape.FillProperty,decisionBrush);
        GainedCount.Text=measured?N(q,"gained").ToString("0"):"—";
        LostCount.Text=measured?N(q,"lost").ToString("0"):"—";
        DecisionReason.Text=reasons?.Count>0?string.Join(". ",reasons.Select(r=>r!.ToString())):
            S(report,"decision",S(state,"readiness","Waiting for a complete comparison."));
        var repl=report?["comparisons"]?["training_effect"]?["repl"]?.AsObject();
        ReplComparisonLabel.Text="REPL comparison";
        if(repl is null && report?["pilot_gate"] is not null)repl=report["comparisons"]?["training_effect"]?["repairs"]?.AsObject();
        if(repl is not null && report?["comparisons"]?["training_effect"]?["repl"] is null)ReplComparisonLabel.Text="Functional repairs";
        ReplComparison.Text=repl is null?"—":
            $"{repl.Sum(r=>N(r.Value,"before")):0} → {repl.Sum(r=>N(r.Value,"after")):0} / {repl.Sum(r=>N(r.Value,"total")):0}";
        var gpu=state["gpu"]; GpuName.Text=S(gpu,"name","GPU information unavailable").Replace("NVIDIA GeForce ","");
        GpuMemory.Text=gpu is null?"":$"{N(gpu,"used_mb")/1024:0.0} / {N(gpu,"total_mb")/1024:0.0} GB VRAM";
        Vram.Value=N(gpu,"total_mb")>0?100*N(gpu,"used_mb")/N(gpu,"total_mb"):0;
        GpuUsage.Text=gpu is null?"":$"{S(gpu,"utilization")}% GPU · {S(gpu,"watts")} W";
        Logs.Text=S(state,"log","No log output yet."); Logs.ScrollToEnd();
        Checkpoints.ItemsSource=state["checkpoints"]!.AsArray().Select(n=>S(n,"path")+" · "+DateTimeOffset.FromUnixTimeSeconds((long)N(n,"saved_at")).ToLocalTime().ToString("g")).DefaultIfEmpty("No complete training checkpoint yet.").ToList();
        LanguageGrid.ItemsSource=q["languages"]!.AsArray().Select(n=>new{Suite=S(n,"suite"),Language=S(n,"language"),Before=N(n,"before"),After=N(n,"after"),Total=N(n,"total"),Gained=n!["gained"]!.AsArray().Count,Lost=n["lost"]!.AsArray().Count}).ToList();
        Chart.Children.Clear();
        var models=state["report"]?["models"]?.AsObject();
        if(models is not null) foreach(var model in models) foreach(var suite in model.Value!.AsObject())
        {
            var panel=new StackPanel{Margin=new Thickness(0,0,0,12)};
            panel.Children.Add(new TextBlock{Text=$"{model.Key.Replace('-',' ')} · {suite.Key}     {N(suite.Value,"passed"):0}/{N(suite.Value,"total"):0}",Margin=new Thickness(0,0,0,5)});
            panel.Children.Add(new ProgressBar{Height=10,Maximum=Math.Max(1,N(suite.Value,"total")),Value=N(suite.Value,"passed")}); Chart.Children.Add(panel);
        }
        ChartHint.Text=report is null?"Waiting for complete matched results. See current task progress above.":"Completed test results. Review language regressions before choosing a checkpoint.";
    }
    void CopyCode_Click(object sender,RoutedEventArgs e)
    {
        try { if(CopyCode.IsEnabled)Clipboard.SetText(displayedCode); }
        catch(Exception ex){Notice.Text="Could not copy code: "+ex.Message;}
    }
    void Theme_Changed(object sender,SelectionChangedEventArgs e)
    {
        if(ThemePicker.SelectedItem is not ComboBoxItem item || item.Content.ToString()==Appearance.Mode)return;
        try
        {
            Appearance.Apply(item.Content.ToString()!,save:true);
            if(latest is not null)Render(latest);
        }
        catch(Exception ex){Notice.Text="Theme could not be saved: "+ex.Message;}
    }
    PowerGpu? SelectedGpu=>PowerDevices.SelectedItem as PowerGpu;
    async Task RefreshPower(bool reset)
    {
        if(powerBusy)return;
        powerBusy=true;
        try
        {
            var devices=await GpuPower.Read();
            var selected=SelectedGpu?.Uuid;
            powerLoading=true;
            PowerDevices.ItemsSource=devices;
            PowerDevices.SelectedItem=devices.FirstOrDefault(g=>g.Uuid==selected)??devices.FirstOrDefault();
            powerLoading=false;
            var gpu=SelectedGpu;
            bool supported=gpu?.Supported==true;
            PowerSlider.IsEnabled=supported; PowerEnable.IsEnabled=supported;
            PowerApply.IsEnabled=PowerRestore.IsEnabled=supported && backend is not null && Session is not null;
            if(gpu is null)throw new IOException("No NVIDIA GPU was found.");
            PowerCurrent.Text=supported?$"Applied now: {gpu.Limit:0.##} W ({100*gpu.Limit/gpu.Default:0.#}%) · Current draw: {gpu.Draw:0.#} W":"This GPU does not expose adjustable power limits.";
            PowerRange.Text=supported?$"Hardware range: {gpu.Minimum:0.##}–{gpu.Maximum:0.##} W. Training presets: {gpu.MinimumPercent}–100% of the {gpu.Default:0.##} W default.":"Power controls are unavailable; training controls still work.";
            var saved=GpuPower.Load();
            bool matches=saved?.Profile==backend?.Profile && saved?.Session==Session && saved?.Uuid==gpu.Uuid;
            PowerSaved.Text=matches?$"Saved for {Session}: {saved!.Percent}%. Idle and paused: 100%.":"No training power limit saved for this experiment and GPU.";
            PowerSlider.Minimum=gpu.MinimumPercent;
            if(reset)
            {
                PowerSlider.Value=matches?saved!.Percent:100;
                PowerPresets.Children.Clear();
                if(supported)for(int percent=100;percent>=gpu.MinimumPercent;percent-=5)
                {
                    int value=percent;
                    var button=new Button{Content=percent+"%"};
                    button.Click+=(_,_)=>PowerSlider.Value=value;
                    PowerPresets.Children.Add(button);
                }
            }
            PowerSlider_Changed(this,null!);
        }
        catch(Exception ex) { PowerFeedback.Text=ex.Message; PowerSlider.IsEnabled=PowerEnable.IsEnabled=PowerApply.IsEnabled=PowerRestore.IsEnabled=false; }
        finally { powerLoading=false;powerBusy=false; }
    }
    async void PowerDevice_Changed(object sender,SelectionChangedEventArgs e){if(!powerLoading)await RefreshPower(true);}
    void PowerSlider_Changed(object sender,RoutedPropertyChangedEventArgs<double> e)
    {
        if(PowerPreview is null || SelectedGpu is not {Supported:true} gpu)return;
        int percent=(int)PowerSlider.Value;
        PowerPreview.Text=$"{percent}% · {gpu.Watts(percent):0.##} W training limit";
    }
    async Task PowerAction(Func<Task> action)
    {
        if(powerBusy)return;
        powerBusy=true; PowerEnable.IsEnabled=PowerApply.IsEnabled=PowerRestore.IsEnabled=false;
        try { await action(); }
        catch(System.ComponentModel.Win32Exception ex) when(ex.NativeErrorCode==1223) { PowerFeedback.Text="Permission request cancelled. The saved limit is unchanged."; }
        catch(Exception ex) { PowerFeedback.Text=ex.Message; }
        finally {powerBusy=false;await RefreshPower(false);}
    }
    async void PowerEnable_Click(object sender,RoutedEventArgs e)=>await PowerAction(async()=>
    {
        await GpuPower.Enable(SelectedGpu!.Uuid);
        PowerFeedback.Text="Power control enabled. Choose a percentage and select Apply.";
    });
    async void PowerApply_Click(object sender,RoutedEventArgs e)=>await SavePower();
    async void PowerRestore_Click(object sender,RoutedEventArgs e){PowerSlider.Value=100;await SavePower();}
    async Task SavePower()=>await PowerAction(async()=>
    {
        if(backend is null || Session is null || SelectedGpu is null)throw new IOException("Connect and select an experiment first.");
        await GpuPower.Save(backend,Session,SelectedGpu.Uuid,(int)PowerSlider.Value);
        PowerFeedback.Text="Saved. The applied limit is confirmed below the GPU selector. Paused work stays at 100%.";
    });
    static string StartupDescription(StartupOptions options) =>
        (options.ResumeLearningOnLogin?"Learning: resumes after login. ":"Learning: start manually. ")+
        (options.OpenStudioOnLogin?"Studio: opens after login.":"Studio: open it yourself.");
    async void Startup_Click(object s,RoutedEventArgs e)=>await Safe(async()=>{
        var options=new StartupOptions(AutoLearning.IsChecked==true,AutoStudio.IsChecked==true);
        await Backend.ConfigureStartup(options);
        StartupSummary.Text="Saved. "+StartupDescription(options);
    });
    async void Connect_Click(object s,RoutedEventArgs e)=>await Safe(Connect);
    async void Detect_Click(object s,RoutedEventArgs e)=>await Safe(async()=>{Distro.ItemsSource=await Backend.Distributions();});
    async void Session_Changed(object s,SelectionChangedEventArgs e){if(!loading) { await Safe(Refresh); await RefreshPower(true); }}
    async void Primary_Click(object s,RoutedEventArgs e)
    {
        var view=ControlView();
        if(busy || !view.Enabled)return;
        if(view.Action=="new"){OpenSettings(0);ExperimentName.Text="coding-model-"+DateTime.Now.ToString("yyyyMMdd-HHmmss");ExperimentName.Focus();return;}
        if(view.Action=="stop"){Stop_Click(s,e);return;}
        if(view.Action=="resume")await StartLearning();
    }
    async Task StartLearning()=>await Safe(async()=>{
        var state=await backend!.Call(new{action="resume",session=Session});
        if(state["worker_alive"]?.GetValue<bool>()!=true) await backend.EnsureWorker(Session!);
        await Refresh();
    });
    async void Pause_Click(object s,RoutedEventArgs e)=>await Safe(async()=>{await backend!.Call(new{action="pause",session=Session});await GpuPower.Sync(backend,Session!,restore:true);await Refresh();});
    async void Stop_Click(object s,RoutedEventArgs e)
    {
        var message=latest?["continuous"]?.GetValue<bool>()==true
            ?"Stop learning?\n\nYour progress stays saved. Learning will stay off after login until you choose Resume. For a gaming break, you can also use Pause."
            :"Stop this learning session?\n\nYour checkpoints and results stay saved. This ends the session. For a gaming break, choose No and use Pause instead.";
        if(MessageBox.Show(this,message,"Stop learning",MessageBoxButton.YesNo,MessageBoxImage.Question)!=MessageBoxResult.Yes)return;
        await Safe(async()=>{await backend!.Call(new{action="stop",session=Session});await GpuPower.Sync(backend,Session!,restore:true);await Refresh();});
    }
    async void Create_Click(object s,RoutedEventArgs e)=>await Safe(async()=>{
        if(backend is null)throw new InvalidOperationException("Connect to WSL first.");
        var name=ExperimentName.Text.Trim();
        await backend.Call(new{action="prepare-loop",name,teacher=TeacherExamples.IsChecked==true},240);
        await LoadSessions(select:name); Tabs.SelectedIndex=0;
    });
    void Export_Click(object s,RoutedEventArgs e)
    {
        if(latest?["report"] is null){Notice.Text="No completed comparison to export yet.";return;}
        var picker=new SaveFileDialog{FileName="training-comparison.json",Filter="JSON report|*.json"};
        if(picker.ShowDialog(this)==true)try{File.WriteAllText(picker.FileName,latest["report"]!.ToJsonString(new JsonSerializerOptions{WriteIndented=true}));Notice.Text="Saved locally. Reports may contain private paths; review before sharing.";}catch(Exception ex){Notice.Text=ex.Message;}
    }
    void Help_Click(object s,RoutedEventArgs e)=>Process.Start(new ProcessStartInfo("https://github.com/M1chlCZ/local-coding-assistant/blob/main/desktop/README.md"){UseShellExecute=true});
}
