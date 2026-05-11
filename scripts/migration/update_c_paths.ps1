[CmdletBinding()]
param(
    [string]$Root,

    [string]$OldProjectRoot,

    [string]$NewProjectRoot,

    [string]$ReplacementMapFile,

    [switch]$IncludeProjectFilesOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDirectory = if ($PSScriptRoot) {
    $PSScriptRoot
} elseif ($MyInvocation.MyCommand.Path) {
    Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    (Get-Location).Path
}

if (-not $Root) {
    $Root = (Resolve-Path (Join-Path $ScriptDirectory '..\..')).Path
}

if (-not $NewProjectRoot) {
    $NewProjectRoot = (Resolve-Path (Join-Path $ScriptDirectory '..\..')).Path
}

function Test-ExcludedPath {
    param(
        [string]$FullPath
    )

    $excludedNames = @(
        'node_modules',
        '__pycache__',
        '.git',
        'venv',
        '.venv',
        '.conda',
        'dist',
        'build',
        'build-rocm',
        '.ipynb_checkpoints',
        '.pytest_cache',
        '.mypy_cache'
    )

    $segments = $FullPath -split '[\\/]'
    foreach ($excluded in $excludedNames) {
        if ($segments -contains $excluded) {
            return $true
        }
    }
    return $false
}

function Test-TextFile {
    param([System.IO.FileInfo]$File)

    $name = $File.Name.ToLowerInvariant()
    $textNames = @('makefile', 'dockerfile')
    $textExtensions = @(
        '.py', '.ps1', '.bat', '.cmd', '.sh', '.json', '.ipynb', '.md', '.txt', '.yml', '.yaml', '.ini', '.env', '.xml', '.toml', '.cfg', '.csv'
    )

    if ($textNames -contains $name) {
        return $true
    }

    foreach ($extension in $textExtensions) {
        if ($name.EndsWith($extension)) {
            return $true
        }
    }

    return $false
}

$Root = ([System.IO.Path]::GetFullPath($Root)).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
$NewProjectRoot = ([System.IO.Path]::GetFullPath($NewProjectRoot)).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
if ($OldProjectRoot) {
    $OldProjectRoot = ([System.IO.Path]::GetFullPath($OldProjectRoot)).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
}

if (-not (Test-Path $Root)) {
    throw "Root does not exist: $Root"
}

$replacementPairs = New-Object System.Collections.Generic.List[object]

$driveLetter = [char]67
$windowsPythonPath = $driveLetter + ':\Python310\python.exe'
$windowsPythonPathAlt = $driveLetter + ':/Python310/python.exe'

if ($OldProjectRoot) {
    $replacementPairs.Add([pscustomobject]@{
        Old = $OldProjectRoot
        New = $NewProjectRoot
    })
}

$replacementPairs.Add([pscustomobject]@{
    Old = $windowsPythonPath
    New = 'python'
})
$replacementPairs.Add([pscustomobject]@{
    Old = $windowsPythonPathAlt
    New = 'python'
})

if ($ReplacementMapFile) {
    if (-not (Test-Path $ReplacementMapFile)) {
        throw "ReplacementMapFile not found: $ReplacementMapFile"
    }

    $rawMap = Get-Content -Path $ReplacementMapFile -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($entry in $rawMap.PSObject.Properties) {
        $replacementPairs.Add([pscustomobject]@{
            Old = [string]$entry.Name
            New = [string]$entry.Value
        })
    }
}

$files = Get-ChildItem -Path $Root -File -Recurse -Force | Where-Object {
    -not (Test-ExcludedPath -FullPath $_.FullName) -and (Test-TextFile -File $_)
}

$total = $files.Count
$changed = 0
$scanned = 0
$filesWithRemainingCPaths = New-Object System.Collections.Generic.List[string]
$pathPattern = $driveLetter + ':\\|'+ $driveLetter + ':/|file:///' + $driveLetter + ':'

Write-Host "Root: $Root"
Write-Host "Files to scan: $total"
Write-Host ""

foreach ($file in $files) {
    $scanned++
    $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
    $originalContent = $content

    foreach ($pair in $replacementPairs) {
        if ([string]::IsNullOrWhiteSpace($pair.Old)) {
            continue
        }
        $content = $content.Replace($pair.Old, $pair.New)
    }

    if ($content -ne $originalContent) {
        $backupPath = "$($file.FullName).bak"
        if (-not (Test-Path $backupPath)) {
            Copy-Item -Path $file.FullName -Destination $backupPath -Force
        }
        Set-Content -Path $file.FullName -Value $content -Encoding UTF8
        $changed++
    }

    if ($content -match $pathPattern) {
        $filesWithRemainingCPaths.Add($file.FullName)
    }

    $percent = [math]::Round(($scanned / $total) * 100, 2)
    Write-Progress -Activity 'Updating path references' -Status "$scanned / $total : $($file.Name)" -PercentComplete $percent
}

Write-Progress -Activity 'Updating path references' -Completed
Write-Host ""
Write-Host "Updated files: $changed"
Write-Host "Scanned files: $scanned"

if ($filesWithRemainingCPaths.Count -gt 0) {
    Write-Host ""
    Write-Host "Files still containing C: references after replacement:"
    $filesWithRemainingCPaths | Sort-Object -Unique | ForEach-Object { Write-Host " - $_" }
    exit 1
}

Write-Host ""
Write-Host "No remaining C: references were found in scanned files."
