@echo off
echo Stopping EEG-to-Text Prediction System
echo ======================================
echo.

:: Define the PID file locations
set "pid_file_backend=%TEMP%\eeg_backend_pid.txt"
set "pid_file_frontend=%TEMP%\eeg_frontend_pid.txt"

:: Stop the backend process
echo Stopping Flask backend server...
if exist "%pid_file_backend%" (
    set /p backend_pid=<"%pid_file_backend%"
    taskkill /F /PID %backend_pid% 2>nul
    if not errorlevel 1 (
        echo Backend server stopped successfully.
    ) else (
        echo No running backend server found.
    )
    del "%pid_file_backend%" 2>nul
) else (
    :: Alternative method if PID file not found
    taskkill /F /FI "WINDOWTITLE eq app.py*" 2>nul
    taskkill /F /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq *Flask*" 2>nul
    echo Attempted to stop backend server.
)

:: Stop the frontend process
echo Stopping React frontend server...
if exist "%pid_file_frontend%" (
    set /p frontend_pid=<"%pid_file_frontend%"
    taskkill /F /PID %frontend_pid% 2>nul
    if not errorlevel 1 (
        echo Frontend server stopped successfully.
    ) else (
        echo No running frontend server found.
    )
    del "%pid_file_frontend%" 2>nul
) else (
    :: Alternative method if PID file not found
    taskkill /F /FI "WINDOWTITLE eq npm*" 2>nul
    taskkill /F /FI "IMAGENAME eq node.exe" 2>nul
    echo Attempted to stop frontend server.
)

:: Kill any remaining processes on the ports
echo Checking for processes on ports 5001 and 5174...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5001"') do (
    taskkill /F /PID %%a 2>nul
    if not errorlevel 1 echo Killed process on port 5001.
)

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5174"') do (
    taskkill /F /PID %%a 2>nul
    if not errorlevel 1 echo Killed process on port 5174.
)

echo.
echo EEG-to-Text Prediction System has been stopped.
echo.
pause
