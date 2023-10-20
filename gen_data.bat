@echo off
setlocal

:: Set the number of times to run the Python file
set "num_runs=99"

:: Set the Python file to run
set "python_file=pure_state_tomography.py"

:: Run the Python file the specified number of times
for /l %%i in (1, 1, %num_runs%) do (
    echo Running iteration %%i
    python %python_file%
)

endlocal