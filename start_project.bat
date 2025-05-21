@echo off
echo Starting EEG-to-Text Prediction System
echo ======================================
echo.

:: Store process IDs in temporary files for later termination
set "pid_file_backend=%TEMP%\eeg_backend_pid.txt"
set "pid_file_frontend=%TEMP%\eeg_frontend_pid.txt"

:: Start the Flask backend server
echo Starting Flask backend server...
start /B cmd /c "cd /d "%~dp0" && python app.py && exit" > backend_log.txt 2>&1

:: Get the PID of the backend process and save it
for /f "tokens=2" %%a in ('tasklist /fi "WINDOWTITLE eq app.py*" /fo list ^| find "PID:"') do (
    echo %%a > "%pid_file_backend%"
)

echo.
echo Waiting for backend to initialize (5 seconds)...
timeout /t 5 /nobreak > nul

echo.
echo Starting React frontend...
cd frontend
start /B cmd /c "npm run dev" > frontend_log.txt 2>&1

:: Get the PID of the frontend process and save it
for /f "tokens=2" %%a in ('tasklist /fi "WINDOWTITLE eq npm*" /fo list ^| find "PID:"') do (
    echo %%a > "%pid_file_frontend%"
)

cd ..

echo.
echo Both servers are now running!
echo - Backend: http://localhost:5001
echo - Frontend: http://localhost:5174
echo.
echo You can now access the application at http://localhost:5174
echo.
echo To stop the servers, run stop_project.bat
echo.
echo Press any key to open the application in your browser...
pause > nul
start http://localhost:5174
