using System.IO;
using System.Windows;
namespace TrainingStudio;
public partial class App : Application
{
    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        if(e.Args.Length==3 && e.Args[0]=="--check-gpu-power")
        {
            try { await GpuPowerCheck.Hardware(e.Args[1],e.Args[2]); Shutdown(0); }
            catch(Exception ex) { File.WriteAllText(e.Args[2]+".error",ex.ToString()); Shutdown(1); }
            return;
        }
        if(e.Args.Length==2 && e.Args[0]=="--enable-gpu-power")
        {
            try { await GpuPower.Install(e.Args[1]); Shutdown(0); }
            catch(Exception ex) { MessageBox.Show(ex.Message,"GPU power setup"); Shutdown(1); }
            return;
        }
        if(e.Args.Length==2 && e.Args[0]=="--self-check")
        {
            try { ControlCheck.ValidateInputs(); File.WriteAllText(e.Args[1],"Passed"); Shutdown(0); }
            catch(Exception ex) { File.WriteAllText(e.Args[1],ex.ToString()); Shutdown(1); }
            return;
        }
        if(e.Args.Length==2 && e.Args[0]=="--check-startup")
        {
            try { await ControlCheck.Startup(e.Args[1]); Shutdown(0); }
            catch(Exception ex) { File.WriteAllText(e.Args[1]+".error",ex.ToString()); Shutdown(1); }
            return;
        }
        if (e.Args.Length == 3 && e.Args[0] == "--worker")
        {
            try { await Backend.Worker(e.Args[1], e.Args[2]); Shutdown(0); }
            catch (Exception ex) { Directory.CreateDirectory(Profile.Home); File.WriteAllText(Path.Combine(Profile.Home,"worker-error.txt"), ex.ToString()); Shutdown(1); }
            return;
        }
        if(e.Args.Length==3 && e.Args[0]=="--check-controls")
        {
            try { await ControlCheck.Run(e.Args[1],e.Args[2]); Shutdown(0); }
            catch(Exception ex) { File.WriteAllText(e.Args[2]+".error",ex.ToString()); Shutdown(1); }
            return;
        }
        Appearance.Initialize();
        var window = new MainWindow(); MainWindow = window; window.Show();
        window.Closed += (_, _) => Shutdown();
        if(e.Args.Length==2 && e.Args[0]=="--check-ui")
        {
            string original=Appearance.Mode;
            try
            {
                if(!await window.Ready.Task.WaitAsync(TimeSpan.FromSeconds(90)))throw new IOException("No live session loaded");
                await System.Windows.Threading.Dispatcher.Yield(System.Windows.Threading.DispatcherPriority.ApplicationIdle);
                window.CheckHomeAccessibility();
                await window.CheckRefreshBehavior();
                foreach(string mode in new[]{"Light","Dark","System"})
                {
                    Appearance.Apply(mode);
                    window.ThemePicker.SelectedIndex=Array.IndexOf(new[]{"System","Light","Dark"},mode);
                    TrainingStudio.MainWindow.CheckThemeContrast();
                    for(int tab=0;tab<window.Tabs.Items.Count;tab++)
                    {
                        window.Tabs.SelectedIndex=tab;
                        await System.Windows.Threading.Dispatcher.Yield(System.Windows.Threading.DispatcherPriority.ApplicationIdle);
                        if(tab==0) {
                            var summary=window.DecisionCard.TransformToAncestor(window.HomeScroll).TransformBounds(new Rect(window.DecisionCard.RenderSize));
                            if(summary.Bottom>window.HomeScroll.ActualHeight+1)throw new IOException("Latest decision is below the default viewport");
                        }
                        Capture(window,e.Args[1]+"."+mode+"."+tab+".png");
                    }
                    for(int page=0;page<window.SettingsTabs.Items.Count;page++) {
                        window.SettingsTabs.SelectedIndex=page;
                        await System.Windows.Threading.Dispatcher.Yield(System.Windows.Threading.DispatcherPriority.ApplicationIdle);
                        Capture(window,e.Args[1]+"."+mode+".settings-"+page+".png");
                        if(page==0) {
                            window.NewSessionScroll.ScrollToEnd();
                            await System.Windows.Threading.Dispatcher.Yield(System.Windows.Threading.DispatcherPriority.ApplicationIdle);
                            var bounds=window.CreateSession.TransformToAncestor(window.NewSessionScroll).TransformBounds(new Rect(window.CreateSession.RenderSize));
                            if(bounds.Top<0 || bounds.Bottom>window.NewSessionScroll.ActualHeight+1)throw new IOException("Create session is not reachable by scrolling");
                            Capture(window,e.Args[1]+"."+mode+".new-session-bottom.png");
                            window.NewSessionScroll.ScrollToTop();
                        }
                    }
                    window.Width=920;window.Height=680;window.Tabs.SelectedIndex=0;
                    await System.Windows.Threading.Dispatcher.Yield(System.Windows.Threading.DispatcherPriority.ApplicationIdle);
                    window.CheckControlStates();
                    Capture(window,e.Args[1]+"."+mode+".compact.png");
                    window.Width=1220;window.Height=880;
                }
                File.WriteAllText(e.Args[1],"State controls, compact layout, background refresh, Light/Dark/System contrast and all settings pages passed. Native backdrop="+window.CheckBackdrop());
                Shutdown(0);
            }
            catch(Exception ex){File.WriteAllText(e.Args[1]+".error",ex.ToString());Shutdown(1);}
            finally{Appearance.Apply(original,save:true);}
            return;
        }
        if(e.Args.Length==2 && e.Args[0]=="--smoke")
        {
            try
            {
                if(!await window.Ready.Task.WaitAsync(TimeSpan.FromSeconds(90))) throw new IOException("No live session loaded");
                await System.Windows.Threading.Dispatcher.Yield(System.Windows.Threading.DispatcherPriority.ApplicationIdle);
                for(int tab=0;tab<window.Tabs.Items.Count;tab++)
                {
                    window.Tabs.SelectedIndex=tab;
                    await System.Windows.Threading.Dispatcher.Yield(System.Windows.Threading.DispatcherPriority.ApplicationIdle);
                    Capture(window,tab==0?e.Args[1]:e.Args[1]+".tab"+tab+".png");
                }
                File.WriteAllText(e.Args[1]+".json", "{\"connected\":true,\"rendered\":true}");
                Shutdown(0);
            }
            catch(Exception ex) { File.WriteAllText(e.Args[1]+".error",ex.ToString()); Shutdown(1); }
        }
    }
    static void Capture(Window window,string path)
    {
        window.UpdateLayout();
        var bitmap=new System.Windows.Media.Imaging.RenderTargetBitmap((int)window.ActualWidth,(int)window.ActualHeight,96,96,System.Windows.Media.PixelFormats.Pbgra32);
        // DWM supplies Mica outside WPF's visual tree; use the theme fallback in offscreen previews.
        var background=new System.Windows.Media.DrawingVisual();
        using(var context=background.RenderOpen()) context.DrawRectangle((System.Windows.Media.Brush)Current.Resources["WindowBrush"],null,new Rect(window.RenderSize));
        bitmap.Render(background);
        bitmap.Render(window);
        var encoder=new System.Windows.Media.Imaging.PngBitmapEncoder();encoder.Frames.Add(System.Windows.Media.Imaging.BitmapFrame.Create(bitmap));
        using var stream=File.Create(path);encoder.Save(stream);
    }

}
