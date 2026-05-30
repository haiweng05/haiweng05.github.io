param(
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "Serving Haiweng Xu homepage at http://127.0.0.1:$Port/"
Write-Host "Press Ctrl+C to stop."
python -m http.server $Port --bind 127.0.0.1
