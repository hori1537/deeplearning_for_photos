@echo off
echo プログラム開始


SET VIRTUAL_ENV_NAME="DLP"

REM 仮想環境をactivate
@echo on
call activate %VIRTUAL_ENV_NAME%

call python deeplearning_for_photos.py