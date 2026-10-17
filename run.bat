@echo off
set PY_EXE=C:\Users\pardh\AppData\Local\Python\pythoncore-3.14-64\python.exe

if exist "%PY_EXE%" (
    "%PY_EXE%" main.py %*
) else (
    python main.py %*
)
