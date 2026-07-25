.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
$env:PORT = "8081"
python -m app.main
