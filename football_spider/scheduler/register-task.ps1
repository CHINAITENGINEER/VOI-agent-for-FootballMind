param([switch]$Register, [switch]$Unregister)
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
$entryPath = Join-Path $projectPath 'platform_cli.py'
$taskName = 'FootballMind-DataPlatform'
if ($Unregister) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Output 'FootballMind task removed; data files retained.'
    exit
}
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument ('"' + $entryPath + '" schedule-tick') -WorkingDirectory $projectPath
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) -RepetitionInterval (New-TimeSpan -Minutes 15)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)
# Interactive user principal is explicit: runs while this user is logged on.
# For unattended logged-off execution use an operator-selected service account.
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'FootballMind durable DB schedule; 15-minute wakeup, source quota-aware jobs.'
$xmlPath = Join-Path $projectPath 'platform_data\FootballMind-DataPlatform.xml'
New-Item -ItemType Directory -Force -Path (Split-Path $xmlPath) | Out-Null
if ($Register) {
    $health = & $pythonPath $entryPath health | ConvertFrom-Json
    if ($health.status -ne 'OK') { throw 'Database health must pass before enabling automatic writes.' }
    Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
    Export-ScheduledTask -TaskName $taskName | Set-Content -LiteralPath $xmlPath -Encoding Unicode
    Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
    Get-ScheduledTaskInfo -TaskName $taskName | Select-Object LastRunTime,LastTaskResult,NextRunTime
} else {
    # Plan-only is intentionally not reported as a registered scheduler.
    [PSCustomObject]@{ TaskName=$taskName; Python=$pythonPath; Entry=$entryPath; WorkingDirectory=$projectPath; WakeupMinutes=15; LogonType='Interactive'; Registered=$false } | ConvertTo-Json
    Write-Output 'CONFIGURATION_GENERATED_ONLY; pass -Register after configuring DATABASE_URL.'
}
