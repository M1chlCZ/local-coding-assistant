using Microsoft.Win32;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Interop;
using System.Text.Json;
using System.Windows;
using System.Windows.Media;
namespace TrainingStudio;

internal static class Appearance
{
    [DllImport("dwmapi.dll")] static extern int DwmSetWindowAttribute(IntPtr window,int attribute,ref int value,int size);
    public static void TitleBar(Window window)
    {
        if(Environment.OSVersion.Version.Build<22000)return;
        var handle=new WindowInteropHelper(window).Handle;
        if(handle==IntPtr.Zero)return;
        foreach(var item in new[]{(35,"WindowBrush"),(36,"TextBrush")}) {
            var c=((SolidColorBrush)Application.Current.Resources[item.Item2]).Color;
            int color=c.R|(c.G<<8)|(c.B<<16);
            DwmSetWindowAttribute(handle,item.Item1,ref color,sizeof(int));
        }
    }
    public static string Mode { get; private set; }="System";
    static string FileName=>Path.Combine(Profile.Home,"appearance.json");
    public static bool IsDark=>Mode=="Dark" || (Mode=="System" && (Registry.GetValue(@"HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize","AppsUseLightTheme",1) is int light && light==0));
    public static void Initialize()
    {
        string mode="System";
        try { if(File.Exists(FileName))mode=JsonSerializer.Deserialize<string>(File.ReadAllText(FileName))??"System"; }
        catch(IOException) { }
        catch(JsonException) { }
        Apply(new[]{"System","Light","Dark"}.Contains(mode)?mode:"System");
        SystemEvents.UserPreferenceChanged+=SystemChanged;
        Application.Current.Exit+=(_,_)=>SystemEvents.UserPreferenceChanged-=SystemChanged;
    }
    static void SystemChanged(object sender,UserPreferenceChangedEventArgs e)
    {
        Application.Current.Dispatcher.BeginInvoke(()=>Apply(Mode));
    }
    internal static Color HighContrastColor(string key)=>key switch {
        "TextBrush" or "MutedBrush" or "LogTextBrush" or "CodeKeywordBrush" or "StepNumberTextBrush" or "BorderBrush" or "PositiveBrush" or "NegativeBrush"=>SystemColors.WindowTextColor,
        "PrimaryBrush"=>SystemColors.HighlightColor,
        "PrimaryTextBrush"=>SystemColors.HighlightTextColor,
        _=>SystemColors.WindowColor
    };
    public static void Apply(string mode,bool save=false)
    {
        if(!new[]{"System","Light","Dark"}.Contains(mode))throw new ArgumentException("Choose System, Light or Dark.");
        if(save)
        {
            Directory.CreateDirectory(Profile.Home);
            File.WriteAllText(FileName+".tmp",JsonSerializer.Serialize(mode));
            File.Move(FileName+".tmp",FileName,true);
        }
        Mode=mode;
        // WPF's native Fluent switch also updates control templates and the title bar.
#pragma warning disable WPF0001
        Application.Current.ThemeMode=mode switch {"Light"=>ThemeMode.Light,"Dark"=>ThemeMode.Dark,_=>ThemeMode.System};
#pragma warning restore WPF0001
        bool dark=IsDark;
        var colors=new (string key,string light,string dark)[]{
            ("BackdropTintBrush","#E0DDE5EE","#CC141D27"),
            ("WindowBrush","#DDE5EE","#141D27"),("CardBrush","#E8E4EBF2","#DE24303D"),
            ("BorderBrush","#B3AABBCD","#805E748D"),("TextBrush","#192637","#D7E1EC"),
            ("MutedBrush","#3B4D62","#B7C6D7"),("NoticeBrush","#E8D6E1ED","#E62B394B"),
            ("InsetBrush","#E8D4DFE9","#DE101C29"),("StepBrush","#E8CCDEF4","#F029435E"),
            ("StepNumberBrush","#5C6D82","#46576C"),("StepNumberTextBrush","#FFFFFF","#E4EBF3"),
            ("CodeKeywordBrush","#064493","#99C2ED"),
            ("ClearBrush","#00E4EBF2","#0024303D"),
            ("LogBrush","#D4DFE9","#101C29"),("LogTextBrush","#24364B","#D0DDEB"),
            ("QualityGoodBrush","#E8D5E7DA","#E51C3C33"),("QualityBadBrush","#E8EBD8D9","#E5442933"),
            ("QualityMixedBrush","#E8EAE0CB","#E53D3528"),("QualityNeutralBrush","#E8D6E1ED","#E52A3748"),
            ("PrimaryBrush","#1455A9","#8AB8EA"),("PrimaryTextBrush","#FFFFFF","#142B43"),
            ("PositiveBrush","#2E7654","#88CBA9"),("NegativeBrush","#B43744","#EB9AA5")};
        foreach(var color in colors)
        {
            var brush=new SolidColorBrush((Color)ColorConverter.ConvertFromString(dark?color.dark:color.light));
            if(SystemParameters.HighContrast) {
                brush=new SolidColorBrush(HighContrastColor(color.key));
            }
            else if(color.key!="ClearBrush" && brush.Color.A<255 && Registry.GetValue(@"HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize","EnableTransparency",1) is int transparency && transparency==0)
                brush.Color=Color.FromRgb(brush.Color.R,brush.Color.G,brush.Color.B);
            brush.Freeze(); Application.Current.Resources[color.key]=brush;
        }
        if(Application.Current.MainWindow is { } window)TitleBar(window);
    }
}
