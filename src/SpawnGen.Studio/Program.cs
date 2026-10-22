using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.SpawnJS;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;
using SpawnGen.Studio;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");
builder.Services.AddSpawnJSRuntime(out _);

// The node types the editor offers: everything marked [Node] in the node assemblies. A node pack is one more AddAssembly.
builder.Services.AddSingleton(new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly));

builder.Services.AddScoped(_ => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });
builder.Services.AddScoped<SpawnGen.Studio.Run.ServerClient>();

await builder.Build().RunAsync();
