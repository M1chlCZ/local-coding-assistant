using System.IO;
using System.Windows;
namespace TrainingStudio;
public partial class App : Application
{
    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
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
        var window = new MainWindow(); MainWindow = window; window.Show();
        window.Closed += (_, _) => Shutdown();
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
                    window.UpdateLayout();
                    var bitmap=new System.Windows.Media.Imaging.RenderTargetBitmap((int)window.ActualWidth,(int)window.ActualHeight,96,96,System.Windows.Media.PixelFormats.Pbgra32);
                    bitmap.Render(window);
                    var encoder=new System.Windows.Media.Imaging.PngBitmapEncoder(); encoder.Frames.Add(System.Windows.Media.Imaging.BitmapFrame.Create(bitmap));
                    using(var stream=File.Create(tab==0?e.Args[1]:e.Args[1]+".tab"+tab+".png")) encoder.Save(stream);
                }
                File.WriteAllText(e.Args[1]+".json", "{\"connected\":true,\"rendered\":true}");
                Shutdown(0);
            }
            catch(Exception ex) { File.WriteAllText(e.Args[1]+".error",ex.ToString()); Shutdown(1); }
        }
    }
}
