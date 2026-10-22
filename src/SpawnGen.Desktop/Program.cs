using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Photino.NET;
using SpawnGen.Server;

// The desktop shell: the same Studio as in a browser, in a native window, with the engine running natively (CUDA / OpenCL).
// The server runs inside this process on a free local port; the window just opens it. Command-line options go to the server
// (--SpawnGen:ModelsPath=... and so on).
internal static class Program
{
    [STAThread]
    private static void Main(string[] args)
    {
        // Loopback only, any free port: nothing on the network can reach it.
        var app = ServerApp.Build(["--urls", "http://127.0.0.1:0", .. args]);
        app.StartAsync().GetAwaiter().GetResult();
        var address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();

        var window = new PhotinoWindow()
            .SetTitle("SpawnGen")
            .SetUseOsDefaultSize(false)
            .SetSize(1600, 950)
            .SetResizable(true)
            .Center()
            .Load(address);
        window.WaitForClose(); // returns when the window is closed

        app.StopAsync().GetAwaiter().GetResult();
        app.DisposeAsync().AsTask().GetAwaiter().GetResult();
    }
}
