using Microsoft.Win32;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Media;
namespace TrainingStudio;

internal static class Appearance
{
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
            ("WindowBrush","#F4F6FA","#151B23"),("CardBrush","#E0FFFFFF","#E0202936"),
            ("BorderBrush","#B3D9E1EC","#B349596D"),("TextBrush","#192637","#EDF2F8"),
            ("MutedBrush","#455568","#B8C6D9"),("NoticeBrush","#E0E9EEF5","#E0283547"),
            ("InsetBrush","#D9EDF2F7","#D9111923"),("StepBrush","#E0DFECFE","#F0253E5B"),
            ("ClearBrush","#00FFFFFF","#00202936"),
            ("LogBrush","#EDF2F7","#111923"),("LogTextBrush","#24364B","#DDE6F3"),
            ("QualityGoodBrush","#E0E7F4EC","#E0163E31"),("QualityBadBrush","#E0FCEDEC","#E04A242A"),
            ("QualityMixedBrush","#E0FFF3DE","#E046361C"),("QualityNeutralBrush","#E0EDF1F7","#E027354A"),
            ("PrimaryBrush","#145BC0","#84B9FF"),("PrimaryTextBrush","#FFFFFF","#092240"),
            ("PositiveBrush","#2E7654","#79D7AA"),("NegativeBrush","#B43744","#FF969F")};
        foreach(var color in colors)
        {
            var brush=new SolidColorBrush((Color)ColorConverter.ConvertFromString(dark?color.dark:color.light));
            if(SystemParameters.HighContrast) {
                brush=new SolidColorBrush(color.key is "TextBrush" or "MutedBrush" or "LogTextBrush" ? SystemColors.WindowTextColor :
                    color.key=="PrimaryBrush"?SystemColors.HighlightColor : color.key=="PrimaryTextBrush"?SystemColors.HighlightTextColor:SystemColors.WindowColor);
            }
            else if(color.key!="ClearBrush" && brush.Color.A<255 && Registry.GetValue(@"HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize","EnableTransparency",1) is int transparency && transparency==0)
                brush.Color=Color.FromRgb(brush.Color.R,brush.Color.G,brush.Color.B);
            brush.Freeze(); Application.Current.Resources[color.key]=brush;
        }
    }
}
