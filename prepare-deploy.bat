@echo off
echo Preparing for Railway deployment...

echo.
echo Checking required files...
if not exist "my_best_model.h5" (
    echo ERROR: my_best_model.h5 not found!
    pause
    exit /b 1
)

if not exist "scaler.pkl" (
    echo ERROR: scaler.pkl not found!
    pause
    exit /b 1
)

if not exist "feature_info.pkl" (
    echo ERROR: feature_info.pkl not found!
    pause
    exit /b 1
)

echo All model files found!

echo.
echo Checking app.py...
if not exist "app.py" (
    echo ERROR: app.py not found!
    pause
    exit /b 1
)

echo.
echo Checking requirements.txt...
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found!
    pause
    exit /b 1
)

echo.
echo Files ready for deployment!
echo.
echo Next steps:
echo 1. Initialize git repository: git init
echo 2. Add files: git add .
echo 3. Commit: git commit -m "Initial commit"
echo 4. Push to GitHub/GitLab
echo 5. Connect repository to Railway
echo.
pause