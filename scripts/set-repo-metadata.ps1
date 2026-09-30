<#
.SYNOPSIS
Sets the GitHub repository description, homepage, and topics that link
previews (Discord, X, Slack) and GitHub search show.

.DESCRIPTION
Run once with GitHub CLI signed in (gh auth login), and again after editing
the values below. The social preview image cannot be set through the API:
upload docs/media/social-preview.png under Settings > General > Social
preview.

.EXAMPLE
.\scripts\set-repo-metadata.ps1
#>
param(
    [string]$Repository = "Acekorneya/POK_Ai"
)

$ErrorActionPreference = "Stop"

$Description = "POK-Agent: a Windows computer-use AI agent. A large model plans, a small fast model acts on screen, and repeated tasks become muscle memory, so the big model is called less every time. Local-first, voice and vision, Rust + Tauri, open source."

$Topics = @(
    "computer-use", "ai-agent", "agentic-ai", "desktop-automation", "windows",
    "llm", "system-1-system-2", "ui-automation", "rpa", "local-first",
    "speech-to-text", "vision-language-model", "rust", "tauri",
    "lm-studio", "ollama", "openrouter", "self-improving-ai", "windows-agent-arena"
)

gh repo edit $Repository --description $Description
if ($LASTEXITCODE -ne 0) { throw "gh repo edit failed; run gh auth login first." }
gh repo edit $Repository --add-topic ($Topics -join ",")
if ($LASTEXITCODE -ne 0) { throw "setting topics failed" }
Write-Host "Description and $($Topics.Count) topics set on $Repository." -ForegroundColor Green
Write-Host "Now upload docs/media/social-preview.png under Settings > General > Social preview." -ForegroundColor Cyan
