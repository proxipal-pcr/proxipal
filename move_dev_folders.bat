@echo off
echo === Moving development resources into developer/ ===

REM Ensure developer folder exists
if not exist developer (
    mkdir developer
)

REM Move each folder using git mv to preserve history
git mv assets developer\assets
git mv docs developer\docs
git mv examples developer\examples
git mv notebooks developer\notebooks
git mv src developer\src
git mv templates developer\templates
git mv tests developer\tests

echo.
echo ✅ All development folders moved into developer/
echo.
echo Next steps:
echo   1. git add .
echo   2. git commit -m "Restructure: move all dev folders into developer/"
echo   3. git push origin main
