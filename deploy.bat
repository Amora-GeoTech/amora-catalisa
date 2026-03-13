@echo off
echo =============================================
echo   AMORA Deploy Script
echo =============================================

set SRC=C:\Users\joaor\Documents\AmoraDigitalRock\Modules\Scripted
set DST=C:\W\AR\Slicer-build\lib\AmoraDigitalRock-5.11\qt-scripted-modules
set SHARE=C:\W\AR\Slicer-build\share\AmoraDigitalRock-5.11
set RES=C:\Users\joaor\Documents\AmoraDigitalRock\Resources

echo.
echo [1/7] Checking paths...
if not exist "%SRC%\AmoraDigitalRock\AmoraDigitalRock.py" (echo ERROR: Source missing! & pause & exit /b 1)
if not exist "%DST%" (echo ERROR: Dest missing! & pause & exit /b 1)

echo [2/7] DELETING ALL .pyc cache files...
del /S /Q "%DST%\*.pyc" 2>nul
echo   Deleted .pyc files

echo [3/7] Cleaning __pycache__ folders...
for /d /r "%DST%" %%d in (__pycache__) do (
    if exist "%%d" rd /s /q "%%d"
)
echo   Cleaned __pycache__

echo [4/7] Copying Python modules...
REM -- AmoraDigitalRock module --
xcopy /Y "%SRC%\AmoraDigitalRock\AmoraDigitalRock.py" "%DST%\"
xcopy /Y "%SRC%\AmoraDigitalRock\AmoraDigitalRockLib\amora_io.py" "%DST%\AmoraDigitalRockLib\"
xcopy /Y "%SRC%\AmoraDigitalRock\AmoraDigitalRockLib\amora_file_reader.py" "%DST%\AmoraDigitalRockLib\"
xcopy /Y "%SRC%\AmoraDigitalRock\AmoraDigitalRockLib\AmoraNetCDFReader.py" "%DST%\AmoraDigitalRockLib\"
xcopy /Y "%SRC%\AmoraDigitalRock\AmoraDigitalRockLib\AmoraNpyReader.py" "%DST%\AmoraDigitalRockLib\"
xcopy /Y "%SRC%\AmoraDigitalRock\AmoraDigitalRockLib\__init__.py" "%DST%\AmoraDigitalRockLib\"
REM -- Home module --
xcopy /Y "%SRC%\Home\Home.py" "%DST%\"
REM -- Filtering module --
xcopy /Y "%SRC%\AmoraFiltering\AmoraFiltering.py" "%DST%\"
REM -- Processing module --
xcopy /Y "%SRC%\AmoraProcessing\AmoraProcessing.py" "%DST%\"
REM -- Simulation module --
xcopy /Y "%SRC%\AmoraSimulation\AmoraSimulation.py" "%DST%\"

echo [5/7] Copying scripts...
xcopy /Y "%SRC%\AmoraFiltering\Resources\Scripts\apply_filter.py" "%DST%\Resources\Scripts\"
xcopy /Y "%SRC%\AmoraProcessing\Resources\Scripts\*.py" "%DST%\Resources\Scripts\"
xcopy /Y "%SRC%\AmoraSimulation\Resources\Scripts\*.py" "%DST%\Resources\Scripts\"

echo [6/8] Copying icons...
xcopy /Y "%SRC%\Home\Resources\Icons\*.png" "%DST%\Resources\Icons\"
xcopy /Y "%SRC%\AmoraDigitalRock\Resources\Icons\*.png" "%DST%\Resources\Icons\"
xcopy /Y "%SRC%\AmoraFiltering\Resources\Icons\*.png" "%DST%\Resources\Icons\"
xcopy /Y "%SRC%\AmoraProcessing\Resources\Icons\*.png" "%DST%\Resources\Icons\"
xcopy /Y "%SRC%\AmoraSimulation\Resources\Icons\*.png" "%DST%\Resources\Icons\"

echo [7/8] Copying stylesheets...
xcopy /Y "%SRC%\Home\Resources\Stylesheets\*.qss" "%DST%\Resources\Stylesheets\"

echo [8/8] Copying volume rendering presets...
if exist "%RES%\VolumeRendering\presets.xml" (
    xcopy /Y "%RES%\VolumeRendering\presets.xml" "%SHARE%\qt-loadable-modules\VolumeRendering\"
    echo   Updated volume rendering presets (rock presets)
) else (
    echo   No custom presets.xml found, skipping
)

echo.
echo =============================================
echo   Verifying key files...
echo =============================================
echo.
echo   AmoraDigitalRock.py sizes:
for %%f in ("%SRC%\AmoraDigitalRock\AmoraDigitalRock.py") do echo     Source: %%~zf bytes
for %%f in ("%DST%\AmoraDigitalRock.py") do echo     Build:  %%~zf bytes
echo   amora_io.py sizes:
for %%f in ("%SRC%\AmoraDigitalRock\AmoraDigitalRockLib\amora_io.py") do echo     Source: %%~zf bytes
for %%f in ("%DST%\AmoraDigitalRockLib\amora_io.py") do echo     Build:  %%~zf bytes
echo   AmoraFiltering.py sizes:
for %%f in ("%SRC%\AmoraFiltering\AmoraFiltering.py") do echo     Source: %%~zf bytes
for %%f in ("%DST%\AmoraFiltering.py") do echo     Build:  %%~zf bytes
echo   apply_filter.py sizes:
for %%f in ("%SRC%\AmoraFiltering\Resources\Scripts\apply_filter.py") do echo     Source: %%~zf bytes
for %%f in ("%DST%\Resources\Scripts\apply_filter.py") do echo     Build:  %%~zf bytes
echo   Home.py sizes:
for %%f in ("%SRC%\Home\Home.py") do echo     Source: %%~zf bytes
for %%f in ("%DST%\Home.py") do echo     Build:  %%~zf bytes

echo.
echo =============================================
echo   DONE! Restart AMORA now.
echo   File sizes should match between Source/Build.
echo =============================================
pause
