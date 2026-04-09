<#
.SYNOPSIS
    Install ao-proxy from git using cargo.

.PARAMETER Repo
    Git repository URL.

.PARAMETER Root
    Cargo install root directory.

.PARAMETER Branch
    Install from a git branch.

.PARAMETER Tag
    Install from a git tag.

.PARAMETER Rev
    Install from a git revision.

.PARAMETER Force
    Reinstall even if already installed.

.DESCRIPTION
    Environment overrides:
      ANTHROPIC_PROXY_INSTALL_REPO
      ANTHROPIC_PROXY_INSTALL_ROOT
      ANTHROPIC_PROXY_INSTALL_BRANCH
      ANTHROPIC_PROXY_INSTALL_TAG
      ANTHROPIC_PROXY_INSTALL_REV
      ANTHROPIC_PROXY_INSTALL_FORCE=1
#>

[CmdletBinding()]
param(
    [string]$Repo,
    [string]$Root,
    [string]$Branch,
    [string]$Tag,
    [string]$Rev,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$DefaultRepoUrl = 'https://github.com/c2js/claudecode-openai-proxy-rs'
$DefaultPackageName = 'ao-proxy'

# Resolve parameters from environment variables if not provided
if (-not $Repo)   { $Repo   = $env:ANTHROPIC_PROXY_INSTALL_REPO   }
if (-not $Repo)   { $Repo   = $DefaultRepoUrl }
if (-not $Root)   { $Root   = $env:ANTHROPIC_PROXY_INSTALL_ROOT   }
if (-not $Branch) { $Branch = $env:ANTHROPIC_PROXY_INSTALL_BRANCH }
if (-not $Tag)    { $Tag    = $env:ANTHROPIC_PROXY_INSTALL_TAG    }
if (-not $Rev)    { $Rev    = $env:ANTHROPIC_PROXY_INSTALL_REV    }
if (-not $Force)  { $Force  = $env:ANTHROPIC_PROXY_INSTALL_FORCE -eq '1' }

# Verify cargo is available
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error 'Missing required command: cargo'
    exit 127
}

# Check that at most one ref is specified
$selectedRefs = @($Branch, $Tag, $Rev) | Where-Object { $_ } | Measure-Object
if ($selectedRefs.Count -gt 1) {
    Write-Error 'Use only one of -Branch, -Tag, or -Rev'
    exit 1
}

$cmd = @('cargo', 'install', '--locked', '--git', $Repo)

if ($Root)   { $cmd += '--root',   $Root   }
if ($Branch) { $cmd += '--branch', $Branch }
elseif ($Tag) { $cmd += '--tag',   $Tag    }
elseif ($Rev) { $cmd += '--rev',   $Rev    }
if ($Force)  { $cmd += '--force'           }

$cmd += $DefaultPackageName

Write-Host "Running: $($cmd -join ' ')"
& $cmd[0] $cmd[1..($cmd.Length - 1)]

if ($LASTEXITCODE -ne 0) {
    Write-Error "cargo install failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

if ($Root) {
    $binaryPath = Join-Path $Root "bin\$DefaultPackageName.exe"
} else {
    $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' }
    $binaryPath = Join-Path $cargoHome "bin\$DefaultPackageName.exe"
}

Write-Host "Installed $DefaultPackageName to $binaryPath"
