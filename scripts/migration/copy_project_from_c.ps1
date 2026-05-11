[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,

    [string]$DestinationRoot,

    [string[]]$ExcludeDirectories = @(
        'node_modules',
        '__pycache__',
        '.git',
        'venv',
        '.venv',
        '.conda',
        'dist',
        'build',
        'build-rocm',
        '.ipynb_checkpoints'
    )
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
    param(
        [string]$FullPath,
        [string[]]$ExcludedNames
    )

    $segments = $FullPath -split '[\\/]'
    foreach ($excluded in $ExcludedNames) {
        if ($segments -contains $excluded) {
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
    New-Item -ItemType Directory -Path $DestinationRoot -Force | Out-Null
}

$files = Get-ChildItem -Path $SourceRoot -File -Recurse -Force | Where-Object {
    -not (Test-ExcludedPath -FullPath $_.FullName -ExcludedNames $ExcludeDirectories)
}

$total = $files.Count
if ($total -eq 0) {
    Write-Host "No files found to copy under $SourceRoot"
    exit 0
}

Write-Host "Source:      $SourceRoot"
Write-Host "Destination: $DestinationRoot"
Write-Host "Files:       $total"
Write-Host ""

$copied = 0
foreach ($file in $files) {
    $relative = $file.FullName.Substring($SourceRoot.Length).TrimStart([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    $targetPath = Join-Path $DestinationRoot $relative
    $targetDir = Split-Path $targetPath -Parent

    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    }

    Copy-Item -Path $file.FullName -Destination $targetPath -Force
    $copied++

    $percent = [math]::Round(($copied / $total) * 100, 2)
    Write-Progress -Activity 'Copying project files' -Status "$copied / $total : $relative" -PercentComplete $percent
}

Write-Progress -Activity 'Copying project files' -Completed
Write-Host ""
Write-Host "Copy completed successfully."
Write-Host "Copied files: $copied"
