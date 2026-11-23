# Runs a command with a RAM watchdog: if free physical memory falls below -MinFreeGB, this run's whole process tree is killed (by PID).
# For heavy GPU / browser runs on NV1080 (2026-11-20: a run that exhausted virtual memory hung the PC).
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-guarded.ps1 -Log _scratch/run.log -MinFreeGB 2 -Command "node drive-site-examples.js" [-WorkDir tools/studio-browser-test]
# Prints the lowest free RAM seen and the command's exit code (3 = killed by the watchdog).
param(
    [Parameter(Mandatory = $true)][string]$Command,
    [string]$Log = "run-guarded.log",
    [double]$MinFreeGB = 2,
    [string]$WorkDir = "."
)
$logPath = [System.IO.Path]::GetFullPath($Log)
$errPath = $logPath + ".err"
$p = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $Command -WorkingDirectory $WorkDir -RedirectStandardOutput $logPath -RedirectStandardError $errPath -PassThru -WindowStyle Hidden
$min = [double]::MaxValue
$killed = $false
while (-not $p.HasExited) {
    Start-Sleep -Seconds 3
    $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    if ($free -lt $min) { $min = $free }
    if ($free -lt $MinFreeGB) {
        $killed = $true
        & taskkill.exe /F /T /PID $p.Id | Out-Null
        break
    }
}
if ($killed) {
    Write-Output ("WATCHDOG KILLED the run: free RAM {0:N1} GB < {1} GB" -f $min, $MinFreeGB)
    exit 3
}
$p.WaitForExit()
Write-Output ("exit {0}; lowest free RAM {1:N1} GB; log {2}" -f $p.ExitCode, $min, $logPath)
exit $p.ExitCode
