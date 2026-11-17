Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "STARTING CRICKET PRESSURE INTELLIGENCE DASHBOARD" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Scope: 352,625 deliveries | 7 Competitions | 1,635 Players" -ForegroundColor Gray
Write-Host "Local URL: http://127.0.0.1:8050" -ForegroundColor Green
Write-Host ""
python "$PSScriptRoot\app.py"
