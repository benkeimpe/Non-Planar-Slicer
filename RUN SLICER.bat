@echo off
cd /d "%~dp0"
echo Starting Non-Planar Spiral Slicer...
echo If your browser does not open, copy the http address shown below into it.
echo.

REM Try the "py" launcher first, then fall back to "python".
py -3 app.py
if not errorlevel 1 goto done
python app.py
if not errorlevel 1 goto done

echo.
echo ====================================================================
echo  Could not start the slicer.
ech