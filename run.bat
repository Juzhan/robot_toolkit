@echo off
setlocal
chcp 65001 >nul
set "PYTHONPATH=%~dp0"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

rem Interpreter: ROBOT_TOOLKIT_PYTHON, or a .robot_toolkit_python file in this
rem directory or one of its parents containing the interpreter path, or
rem py/python from PATH.  Point it at an environment that has the optional
rem deps you need (lerobot for "export", pyAgxArm + python-can for real CAN).
rem Nothing here is machine-specific: the pointer file is gitignored.
if defined ROBOT_TOOLKIT_PYTHON goto have_python
for %%F in (
  "%~dp0.robot_toolkit_python"
  "%~dp0..\.robot_toolkit_python"
  "%~dp0..\..\.robot_toolkit_python"
) do (
  if not defined ROBOT_TOOLKIT_PYTHON if exist "%%~fF" set /p ROBOT_TOOLKIT_PYTHON=<"%%~fF"
)
if defined ROBOT_TOOLKIT_PYTHON goto have_python

where py >nul 2>nul
if not errorlevel 1 (
  set "ROBOT_TOOLKIT_PYTHON=py"
  goto have_python
)
where python >nul 2>nul
if not errorlevel 1 (
  set "ROBOT_TOOLKIT_PYTHON=python"
  goto have_python
)

echo 未找到 Python。
echo 请设置 ROBOT_TOOLKIT_PYTHON，或创建 .robot_toolkit_python 文件写入解释器路径。
pause
exit /b 1

:have_python
if exist "%ROBOT_TOOLKIT_PYTHON%" (
  "%ROBOT_TOOLKIT_PYTHON%" -X utf8 "%~dp0run.py" %*
) else (
  %ROBOT_TOOLKIT_PYTHON% -X utf8 "%~dp0run.py" %*
)
exit /b %ERRORLEVEL%
