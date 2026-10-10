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
namespace TrainingStudio;

public partial class MainWindow : Window
{
    Backend? backend;
    JsonNode? latest;
    bool busy, loading, powerBusy, powerLoading;
    readonly DispatcherTimer timer = new() { Interval=TimeSpan.FromSeconds(5) };
    public TaskCompletionSource<bool> Ready { get; } = new();
    string? Session => SessionPicker.SelectedItem as string;
    public MainWindow()
    {
        InitializeComponent(); Controls.IsEnabled=false;
        try { var p=Profile.Load(); Distro.Text=p.Distribution; LinuxUser.Text=p.User; ProjectRoot.Text=p.Root; PythonPath.Text=p.Python; }
        catch(Exception ex) { Notice.Text="Saved connection could not be read: "+ex.Message; }
        try { var options=StartupOptions.Load(); AutoLearning.IsChecked=options.ResumeLearningOnLogin; AutoStudio.IsChecked=options.OpenStudioOnLogin; StartupSummary.Text=StartupDescription(options); }
        catch(Exception ex) { Notice.Text="Startup preferences could not be read: "+ex.Message; }
        Loaded += async (_,_) => { await Safe(async()=>{if(!string.IsNullOrWhiteSpace(ProjectRoot.Text)) await Connect(); else Tabs.SelectedIndex=6;}); await RefreshPower(true); Ready.TrySetResult(latest is not null); timer.Start(); };
        timer.Tick += async (_,_) => {if(!busy && backend is not null && Session is not null) await Safe(Refresh,false); if(PowerTab.IsSelected && !powerBusy) await RefreshPower(false);};
        Closed += (_,_)=>timer.Stop();
    }
    async Task Safe(Func<Task> work, bool announce=true)
    {
        if(busy) return;
        busy=true; Controls.IsEnabled=false; SessionPicker.IsEnabled=false;
        try { await work(); if(announce) Notice.Text="Ready. Data and model files stay on this PC."; }
        catch(Exception ex) { latest=null; Notice.Text=ex.Message; ConnectionBadge.Text="Action failed · check details below"; if(!announce) {Activity.Text="Connection unavailable"; ActivityDetail.Text="Displayed results are from the last successful read. Training may still be running."; StageProgress.IsIndeterminate=false;} }
        finally { busy=false; SessionPicker.IsEnabled=true; UpdateControls(); }
    }
    void UpdateControls()
    {
        Controls.IsEnabled=backend is not null && latest is not null && Session is not null;
        string status=latest?["status"]?.ToString()??"";
        bool terminal=new[]{"completed","stopped","budget_exhausted"}.Contains(status);
        ResumeButton.IsEnabled=!terminal;
        PauseButton.IsEnabled=!terminal && status!="paused";
        StopButton.IsEnabled=!terminal;
    }
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
        else {Activity.Text="No experiments yet"; ActivityDetail.Text="Prepare a new experiment from the New experiment tab."; Tabs.SelectedIndex=3;}
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
            row.Children.Add(new ProgressBar{Minimum=0,Maximum=100,Value=Math.Abs(item.Delta),Height=10,Foreground=item.Delta<0?Brushes.IndianRed:Brushes.SeaGreen});
            HistoryChart.Children.Add(row);
        }
    }
    async Task Refresh()
    {
        if(Session is null || backend is null) return;
        bool hadReport=latest?["report"] is not null;
        latest=await backend.Call(new{action="status",session=Session}); Render(latest);
        if(!hadReport && latest["report"] is not null) RenderHistory(await backend.Call(new{action="list"}));
        ConnectionBadge.Text="Updated "+DateTime.Now.ToString("HH:mm:ss")+" · local WSL";
    }
    static string S(JsonNode? n,string key,string fallback="—")=>n?[key]?.ToString()??fallback;
    static double N(JsonNode? n,string key)=>double.TryParse(n?[key]?.ToString(),NumberStyles.Float,CultureInfo.InvariantCulture,out var v)?v:0;
    static string Duration(double seconds)=>TimeSpan.FromSeconds(Math.Max(0,seconds)).ToString(@"hh\:mm\:ss");
    void Render(JsonNode state)
    {
        var q=state["quality"]!;
        QualityTitle.Text=S(q,"label");
        string color=S(q,"tone") switch {"good"=>"#E7F4EC","bad"=>"#FCEDEC","mixed"=>"#FFF3DE",_=>"#EDF1F7"};
        QualityCard.Background=(Brush)new BrushConverter().ConvertFromString(color)!;
        QualityDetail.Text=N(q,"total")>0?$"{N(q,"before"):0} → {N(q,"after"):0} passed out of {N(q,"total"):0}. Newly solved: {N(q,"gained"):0}. Newly failed: {N(q,"lost"):0}.":"Complete the baseline and trained benchmarks to measure a change.";
        Activity.Text=S(state,"status").Replace('_',' ') + " · " + S(state,"phase").Replace('-',' ');
        ActivityDetail.Text=S(state,"detail");
        var p=state["stage_progress"];
        double done=N(p,"completed"),total=N(p,"total");
        if(total<=0){done=N(p,"step");total=N(p,"max_steps");}
        StageProgress.IsIndeterminate=total<=0 && S(state,"status")=="running";
        StageProgress.Value=total>0?100*done/total:0;
        ProgressText.Text=$"Stage {Math.Min(N(state,"stage_index")+1,N(state,"stage_total")):0}/{N(state,"stage_total"):0}"+(total>0?$" · {done:0}/{total:0}":"")+" · "+S(p,"phase","")+" "+S(p,"language","");
        if(p?["task"] is not null) ProgressText.Text+=" · "+S(p,"task");
        if(N(p,"heartbeat_at")>0 && DateTimeOffset.UtcNow.ToUnixTimeSeconds()-N(p,"heartbeat_at")>120) ProgressText.Text+=" · No stage update for over 2 minutes; inspect the log";
        Clocks.Text="Training "+Duration(N(state,"training_seconds"))+" / "+Duration(N(state,"limit_seconds"))+" · All active work "+Duration(N(state,"active_seconds"));
        var gpu=state["gpu"]; GpuName.Text=S(gpu,"name","GPU information unavailable");
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
        ChartHint.Text=models is null?"Waiting for complete matched results. See current task progress above.":"Completed test results. Review language regressions before choosing a checkpoint.";
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
    async void Resume_Click(object s,RoutedEventArgs e)=>await Safe(async()=>{
        var state=await backend!.Call(new{action="resume",session=Session});
        if(state["worker_alive"]?.GetValue<bool>()!=true) await backend.EnsureWorker(Session!);
        await Refresh();
    });
    async void Pause_Click(object s,RoutedEventArgs e)=>await Safe(async()=>{await backend!.Call(new{action="pause",session=Session});await GpuPower.Sync(backend,Session!,restore:true);await Refresh();});
    async void Stop_Click(object s,RoutedEventArgs e)
    {
        if(MessageBox.Show(this,"End this experiment? Its checkpoints and results remain saved. Stop is final; choose Pause if you want to resume later.","Stop experiment",MessageBoxButton.YesNo,MessageBoxImage.Question)!=MessageBoxResult.Yes)return;
        await Safe(async()=>{await backend!.Call(new{action="stop",session=Session});await GpuPower.Sync(backend,Session!,restore:true);await Refresh();});
    }
    async void ValidateData_Click(object s,RoutedEventArgs e)=>await Safe(async()=>{
        if(backend is null)throw new InvalidOperationException("Connect to WSL first.");
        var data=await backend.Call(new{action="validate-data",dataset=DatasetPath.Text.Trim()},180);
        DatasetPreview.Text=$"{S(data,"rows")} examples · {data["languages"]}\nEvidence: {S(data,"verification")}\n\nPreview: {S(data,"preview")}";
    });
    async void Create_Click(object s,RoutedEventArgs e)=>await Safe(async()=>{
        if(backend is null)throw new InvalidOperationException("Connect to WSL first.");
        var name=ExperimentName.Text.Trim();
        var hours=int.Parse(((ComboBoxItem)Hours.SelectedItem).Tag.ToString()!,CultureInfo.InvariantCulture);
        await backend.Call(new{action="prepare",name,dataset=DatasetPath.Text.Trim(),hours},240);
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
