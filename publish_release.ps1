# Publishes the v15 run to https://github.com/KHAYAATI/ADAPT-PHISH_2026
# Usage (PowerShell, from this folder):
#   powershell -ExecutionPolicy Bypass -File publish_release.ps1 -RunDir "C:\...\adapt_phish_v15_gpu_results"
param([Parameter(Mandatory=$true)][string]$RunDir,
      [string]$Remote = "https://github.com/KHAYAATI/ADAPT-PHISH_2026.git")
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here
# RunDir may be: the run output folder (containing release\), the release folder itself, or release.zip
$Src = $null
if ($RunDir -like "*.zip" -and (Test-Path $RunDir)) {
    $Src = Join-Path $env:TEMP "adapt_phish_release_unzipped"
    if (Test-Path $Src) { Remove-Item -Recurse -Force $Src }
    Expand-Archive -Path $RunDir -DestinationPath $Src
} elseif (Test-Path (Join-Path $RunDir "README_RELEASE.md")) { $Src = $RunDir }
elseif (Test-Path (Join-Path $RunDir "release\README_RELEASE.md")) { $Src = Join-Path $RunDir "release" }
elseif (Test-Path (Join-Path $RunDir "release.zip")) {
    $Src = Join-Path $env:TEMP "adapt_phish_release_unzipped"
    if (Test-Path $Src) { Remove-Item -Recurse -Force $Src }
    Expand-Archive -Path (Join-Path $RunDir "release.zip") -DestinationPath $Src
}
if (-not $Src -or -not (Test-Path (Join-Path $Src "README_RELEASE.md"))) {
    Write-Host "No release package found in $RunDir. Searching your user folder for it..."
    Get-ChildItem $env:USERPROFILE -Recurse -File -Include README_RELEASE.md, release.zip -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName
    throw "Re-run with -RunDir set to one of the folders (or release.zip) listed above."
}

# 1) copy the release package (checkpoints/ and cache/ are not part of it)
if (Test-Path "release") { Remove-Item -Recurse -Force "release" }
Copy-Item -Recurse $Src "release"

# 2) remove the audit answer key wherever it was copied (R14 copies revision/ into release/results/)
$key = Get-ChildItem -Recurse -File "release" | Where-Object { $_.Name -like "*audit_key*" }
foreach ($k in $key) { Write-Host "removed (must stay private):" $k.FullName; Remove-Item -Force $k.FullName }
# SHA256SUMS.txt listed the key: rewrite it without that line
$sums = "release\SHA256SUMS.txt"
if (Test-Path $sums) { (Get-Content $sums) | Where-Object { $_ -notmatch "audit_key" } | Set-Content -Encoding utf8 $sums }

# 3) safety checks
$bad = Get-ChildItem -Recurse -File . | Where-Object { $_.FullName -match "\\(checkpoints|cache)\\" -or $_.Name -like "*audit_key*" }
if ($bad) { $bad | ForEach-Object { $_.FullName }; throw "Forbidden files present: stop." }
$big = Get-ChildItem -Recurse -File . | Where-Object { $_.FullName -notmatch "\\.git\\" -and $_.Length -gt 95MB }
if ($big) { $big | ForEach-Object { "{0}  {1:N0} MB" -f $_.FullName, ($_.Length/1MB) }; throw "Files above GitHub's 100 MB limit: deposit them on Zenodo or Hugging Face and delete them here." }
$mb = (Get-ChildItem -Recurse -File . | Where-Object { $_.FullName -notmatch "\\.git\\" } | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Files ready: {0:N0} MB" -f $mb)

# 4) commit and push (asks for your GitHub login or token the first time)
if (-not (Test-Path ".git")) { git init -b main | Out-Null; git remote add origin $Remote }
git add -A
git status --short | Select-Object -First 40
$ok = Read-Host "Push these files to $Remote ? (yes/no)"
if ($ok -ne "yes") { Write-Host "Nothing pushed."; exit 0 }
git commit -m "ADAPT-PHISH v15: full GPU run, release package, paper scripts, verification and blind audit sheet"
git push -u origin main
git log -1 --format="Pushed commit %h on %ci"
