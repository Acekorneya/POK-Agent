# Runs one Windows Agent Arena task with POK-Agent inside the arena VM.
# Started (detached) by the POK-Agent WAA agent; see scripts/arena/waa_agent/agent.py.
#
# Inputs, uploaded by the agent to C:\pokai\runs\<RunId>\:
#   task.txt          the task instruction
#   memory_scope.txt  memory to reuse from the share ("" = fresh memory)
#   keys.env          NAME=value API keys; read into this process, then deleted
# Outputs, written to \\host.lan\Data\pokai\out\<RunId>\:
#   log.txt (stdout), stderr.txt, trace.jsonl, observation JSON/OCR, curation JSON,
#   router-training.jsonl, exit.txt (written last)
# Memory is saved to \\host.lan\Data\pokai\memory\<scope>\ when a scope is set.
param([Parameter(Mandatory = $true)][string]$RunId)

$ErrorActionPreference = "Continue"
$share = "\\host.lan\Data\pokai"
$root = "C:\pokai"
$run = Join-Path $root "runs\$RunId"
$out = Join-Path $share "out\$RunId"
$data = Join-Path $root "data"
New-Item -ItemType Directory -Force -Path $run, $out, $data | Out-Null
$exitCode = 1

try {
    # This VM is a disposable test machine. Defender's heuristics have flagged
    # new builds of this unsigned automation binary (it drives other apps and
    # simulates input) as potentially unwanted and blocked every later start,
    # so the agent's own folder is excluded here, in the test VM only.
    try { Add-MpPreference -ExclusionPath $root -ErrorAction Stop } catch { }

    # The binary and configuration for this run.
    Copy-Item "$share\bin\*" $root -Force

    # Memory: restore the named scope, or start empty.
    $scope = (Get-Content "$run\memory_scope.txt" -Raw -ErrorAction SilentlyContinue)
    $scope = if ($scope) { $scope.Trim() } else { "" }
    Remove-Item "$data\memory" -Recurse -Force -ErrorAction SilentlyContinue
    if ($scope -and (Test-Path "$share\memory\$scope")) {
        robocopy "$share\memory\$scope" "$data\memory" /E /NFL /NDL /NJH /NJS /NP | Out-Null
    }

    # Keys live only in this process's environment.
    if (Test-Path "$run\keys.env") {
        foreach ($line in Get-Content "$run\keys.env") {
            $name, $value = $line -split "=", 2
            if ($name -and $value) { [Environment]::SetEnvironmentVariable($name.Trim(), $value, "Process") }
        }
        Remove-Item "$run\keys.env" -Force
    }

    $prompt = Get-Content "$run\task.txt" -Raw
    $before = @(Get-ChildItem "$root\diag\sessions" -Directory -ErrorAction SilentlyContinue | ForEach-Object Name)
    # Windows PowerShell 5.1 does not escape embedded quotes when it passes an
    # argument to a native program, so build the command line explicitly
    # (backslashes before a quote are doubled, the quote is escaped).
    $quoted = '"' + (($prompt -replace '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"'
    $process = Start-Process -FilePath "$root\pok-ai.exe" -NoNewWindow -PassThru `
        -ArgumentList "--config `"$root\arena.toml`" run $quoted --yes" `
        -RedirectStandardOutput "$out\log.txt" -RedirectStandardError "$out\stderr.txt"
    $null = $process.Handle  # keeps the exit code readable after exit
    # Waits for pok-ai.exe itself only; a browser it started may keep running.
    $process.WaitForExit()
    $exitCode = $process.ExitCode

    # This run's session: diagnostics without screenshots (text evidence only).
    $session = Get-ChildItem "$root\diag\sessions" -Directory -ErrorAction SilentlyContinue |
        Where-Object { $before -notcontains $_.Name } | Sort-Object LastWriteTime | Select-Object -Last 1
    if ($session) {
        Get-ChildItem $session.FullName -File |
            Where-Object { $_.Name -like "*.jsonl" -or $_.Name -like "*.json" -or $_.Name -like "*-ocr.txt" } |
            Copy-Item -Destination $out -Force
        $training = Join-Path $data "router-training\$($session.Name).jsonl"
        if (Test-Path $training) { Copy-Item $training "$out\router-training.jsonl" -Force }
    }

    # Save memory back so later tasks and passes in this scope can use it.
    if ($scope) {
        New-Item -ItemType Directory -Force -Path "$share\memory\$scope" | Out-Null
        robocopy "$data\memory" "$share\memory\$scope" /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
    }
}
catch {
    $_ | Out-String | Add-Content "$out\log.txt"
}
finally {
    Remove-Item "$run\keys.env" -Force -ErrorAction SilentlyContinue
    Set-Content -Path "$out\exit.txt" -Value $exitCode
}
