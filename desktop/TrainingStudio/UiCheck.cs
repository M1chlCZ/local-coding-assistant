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
        var saved=latest;
        try {
            foreach(string status in new[]{"running","paused","stopped","completed","interrupted","failed","pausing","stopping","waiting"}) {
                latest=saved!.DeepClone();latest["status"]=status;latest["desired"]=status is "paused" or "pausing"?"pause":status is "stopped" or "stopping"?"stop":"resume";
                latest["worker_alive"]=status is not("interrupted" or "stopped" or "completed");latest["active_seconds"]=10;
                UpdateControls();UpdateLayout();
                if(PrimaryText.Text!=ControlView().Button || StatusTitle.Text!=ControlView().Title)throw new IOException("State text differs from controls");
                var position=PrimaryButton.TranslatePoint(new Point(),this);
                if(position.X<0 || position.X+PrimaryButton.ActualWidth>ActualWidth)throw new IOException("Main control is clipped");
            }
        } finally {latest=saved;UpdateControls();}
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
        double Light(string key)
        {
            var c=((SolidColorBrush)Application.Current.Resources[key]).Color;
            double Channel(byte b){double v=b/255.0;return v<=0.04045?v/12.92:Math.Pow((v+0.055)/1.055,2.4);}
            return 0.2126*Channel(c.R)+0.7152*Channel(c.G)+0.0722*Channel(c.B);
        }
        void Check(string text,string background)
        {
            double a=Light(text),b=Light(background);
            if((Math.Max(a,b)+0.05)/(Math.Min(a,b)+0.05)<4.5)throw new IOException($"Low theme contrast: {text} on {background}");
        }
        foreach(var background in new[]{"WindowBrush","CardBrush","NoticeBrush","QualityGoodBrush","QualityBadBrush","QualityMixedBrush","QualityNeutralBrush"})
        {
            Check("TextBrush",background);Check("MutedBrush",background);
        }
        Check("LogTextBrush","LogBrush");
        Check("PrimaryTextBrush","PrimaryBrush");
    }
}
