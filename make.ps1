# PowerShell build script for Virtual Try-On Project
# Alternative to Makefile for Windows users

param(
    [Parameter(Position=0)]
    [string]$Target = "help"
)

$VENV_DIR = ".venv"
$PYTHON_VENV = "$VENV_DIR\Scripts\python.exe"
$PIP_VENV = "$VENV_DIR\Scripts\pip.exe"
$STREAMLIT_VENV = "$VENV_DIR\Scripts\streamlit.exe"

function Show-Help {
    Write-Host "Virtual Try-On - Available targets:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  .\make.ps1 setup     - Create virtual environment and install dependencies"
    Write-Host "  .\make.ps1 install   - Install/update dependencies"
    Write-Host "  .\make.ps1 run       - Run the Streamlit application"
    Write-Host "  .\make.ps1 test      - Run pytest tests"
    Write-Host "  .\make.ps1 lint      - Run code linting (flake8)"
    Write-Host "  .\make.ps1 format    - Format code with black"
    Write-Host "  .\make.ps1 clean     - Remove generated files and caches"
    Write-Host "  .\make.ps1 dev       - Install development dependencies"
    Write-Host "  .\make.ps1 check     - Run lint and test"
    Write-Host ""
}

function Invoke-Setup {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $VENV_DIR
    
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    & $PIP_VENV install --upgrade pip
    & $PIP_VENV install -r requirements.txt
    
    Write-Host "Setup complete! Activate with: $VENV_DIR\Scripts\Activate.ps1" -ForegroundColor Green
}

function Invoke-Install {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    & $PIP_VENV install --upgrade pip
    & $PIP_VENV install -r requirements.txt
    Write-Host "Dependencies installed!" -ForegroundColor Green
}

function Invoke-Dev {
    Invoke-Install
    Write-Host "Installing development dependencies..." -ForegroundColor Yellow
    & $PIP_VENV install pytest pytest-cov black flake8 mypy isort
    Write-Host "Dev dependencies installed!" -ForegroundColor Green
}

function Invoke-Run {
    Write-Host "Starting Streamlit application..." -ForegroundColor Yellow
    & $STREAMLIT_VENV run app.py
}

function Invoke-Test {
    Write-Host "Running tests..." -ForegroundColor Yellow
    & $PYTHON_VENV -m pytest -v
}

function Invoke-Lint {
    Write-Host "Running flake8..." -ForegroundColor Yellow
    & $PYTHON_VENV -m flake8 src/ app.py --max-line-length=100 --extend-ignore=E203,W503
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Linting passed!" -ForegroundColor Green
    }
}

function Invoke-Format {
    Write-Host "Formatting code with black..." -ForegroundColor Yellow
    & $PYTHON_VENV -m black src/ app.py --line-length=100
    
    Write-Host "Sorting imports with isort..." -ForegroundColor Yellow
    & $PYTHON_VENV -m isort src/ app.py --profile black
    
    Write-Host "Formatting complete!" -ForegroundColor Green
}

function Invoke-Clean {
    Write-Host "Cleaning generated files..." -ForegroundColor Yellow
    
    $patterns = @(
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "htmlcov",
        ".coverage",
        "dist",
        "build",
        "*.egg-info"
    )
    
    foreach ($pattern in $patterns) {
        Get-ChildItem -Path . -Recurse -Filter $pattern -Force -ErrorAction SilentlyContinue | 
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    Write-Host "Clean complete!" -ForegroundColor Green
}

function Invoke-Check {
    Write-Host "Running all checks..." -ForegroundColor Yellow
    Invoke-Lint
    Invoke-Test
    Write-Host "All checks complete!" -ForegroundColor Green
}

# Main script logic
switch ($Target.ToLower()) {
    "help" { Show-Help }
    "setup" { Invoke-Setup }
    "install" { Invoke-Install }
    "dev" { Invoke-Dev }
    "run" { Invoke-Run }
    "test" { Invoke-Test }
    "lint" { Invoke-Lint }
    "format" { Invoke-Format }
    "clean" { Invoke-Clean }
    "check" { Invoke-Check }
    default {
        Write-Host "Unknown target: $Target" -ForegroundColor Red
        Show-Help
        exit 1
    }
}
