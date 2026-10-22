using SpawnGen.Server;

// SpawnGen server: the ComfyUI-shaped API plus the Studio, one process. Configure with appsettings or SpawnGen__ModelsPath etc.
await ServerApp.Build(args).RunAsync();
