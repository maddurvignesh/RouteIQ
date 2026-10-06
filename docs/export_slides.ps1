# Export every slide of the deck to PNG (Windows + PowerPoint only).
#
#   powershell -ExecutionPolicy Bypass -File docs\export_slides.ps1
#
# Useful for proof-reading the layout without opening PowerPoint. The PNGs land
# in docs\_slides\slide\SlideN.PNG. Delete the folder afterwards - it is a
# preview artefact, not part of the deliverable.

param(
    [string]$Deck = "docs\Adaptive_Routing_Presentation.pptx",
    [string]$OutDir = "docs\_slides"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Deck)) {
    Write-Error "Deck not found: $Deck - run 'python docs/build_presentation.py' first."
    exit 1
}

$deckPath = (Resolve-Path $Deck).Path
if (Test-Path $OutDir) { Remove-Item $OutDir -Recurse -Force }
New-Item -ItemType Directory -Path $OutDir | Out-Null
$outPath = (Resolve-Path $OutDir).Path

$pp = New-Object -ComObject PowerPoint.Application
try {
    # Open read-only, untitled, without a window: 18 = ppSaveAsPNG
    $pres = $pp.Presentations.Open($deckPath, $true, $false, $false)
    $pres.SaveCopyAs((Join-Path $outPath "slide"), 18, $false)
    $pres.Close()
    Write-Output "Exported $($pres.Slides.Count) slides to $outPath"
} finally {
    $pp.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pp) | Out-Null
}

Get-ChildItem -Path $outPath -Recurse -Filter *.PNG | ForEach-Object { $_.FullName }
