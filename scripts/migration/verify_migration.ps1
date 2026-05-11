[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,

    [string]$DestinationRoot,

    [switch]$ShowRemainingCPaths
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

if (-not $DestinationRoot) {
    $DestinationRoot = (Resolve-Path (Join-Path $ScriptDirectory '..\..')).Path
}

function Test-ExcludedPath {
    param([string]$FullPath)

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

$SourceRoot = ([System.IO.Path]::GetFullPath($SourceRoot)).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
$DestinationRoot = ([System.IO.Path]::GetFullPath($DestinationRoot)).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)

if (-not (Test-Path $SourceRoot)) {
    throw "SourceRoot does not exist: $SourceRoot"
}

if (-not (Test-Path $DestinationRoot)) {
    throw "DestinationRoot does not exist: $DestinationRoot"
}

$sourceFiles = Get-ChildItem -Path $SourceRoot -File -Recurse -Force | Where-Object {
    -not (Test-ExcludedPath -FullPath $_.FullName) -and (Test-TextFile -File $_)
}
$destinationFiles = Get-ChildItem -Path $DestinationRoot -File -Recurse -Force | Where-Object {
    -not (Test-ExcludedPath -FullPath $_.FullName) -and (Test-TextFile -File $_)
}

$sourceLookup = @{}
foreach ($file in $sourceFiles) {
    $relative = $file.FullName.Substring($SourceRoot.Length).TrimStart([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    $sourceLookup[$relative] = $file
}

$destinationLookup = @{}
foreach ($file in $destinationFiles) {
    $relative = $file.FullName.Substring($DestinationRoot.Length).TrimStart([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    $destinationLookup[$relative] = $file
}

$missing = New-Object System.Collections.Generic.List[string]
$extra = New-Object System.Collections.Generic.List[string]
$mismatchedSize = New-Object System.Collections.Generic.List[string]
$driveLetter = [char]67
$pathPattern = $driveLetter + ':\\|'+ $driveLetter + ':/|file:///' + $driveLetter + ':'

foreach ($relative in $sourceLookup.Keys) {
    if (-not $destinationLookup.ContainsKey($relative)) {
        $missing.Add($relative)
        continue
    }

    $sourceItem = $sourceLookup[$relative]
    $destinationItem = $destinationLookup[$relative]
    if ($sourceItem.Length -ne $destinationItem.Length) {
        $mismatchedSize.Add($relative)
    }
}

foreach ($relative in $destinationLookup.Keys) {
    if (-not $sourceLookup.ContainsKey($relative)) {
        $extra.Add($relative)
    }
}

Write-Host "Source files:      $($sourceFiles.Count)"
Write-Host "Destination files: $($destinationFiles.Count)"
Write-Host "Missing files:     $($missing.Count)"
Write-Host "Extra files:       $($extra.Count)"
Write-Host "Size mismatches:   $($mismatchedSize.Count)"

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Missing relative paths:"
    $missing | Sort-Object | ForEach-Object { Write-Host " - $_" }
}

if ($extra.Count -gt 0) {
    Write-Host ""
    Write-Host "Unexpected extra relative paths:"
    $extra | Sort-Object | ForEach-Object { Write-Host " - $_" }
}

if ($mismatchedSize.Count -gt 0) {
    Write-Host ""
    Write-Host "Files with different sizes:"
    $mismatchedSize | Sort-Object | ForEach-Object { Write-Host " - $_" }
}

$remainingCRefs = @()
if ($ShowRemainingCPaths) {
    $scanFiles = $destinationFiles | Where-Object { Test-TextFile -File $_ }
    foreach ($file in $scanFiles) {
        $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
        if ($content -match $pathPattern) {
            $remainingCRefs += $file.FullName
        }
    }

    Write-Host ""
    Write-Host "Files still containing C: references: $($remainingCRefs.Count)"
    $remainingCRefs | Sort-Object -Unique | ForEach-Object { Write-Host " - $_" }
}

if ($missing.Count -gt 0 -or $extra.Count -gt 0 -or $mismatchedSize.Count -gt 0) {
    exit 1
}

Write-Host ""
Write-Host "Migration verification passed."
