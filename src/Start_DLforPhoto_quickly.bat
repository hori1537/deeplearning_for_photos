@echo off
echo �v���O�����J�n


SET VIRTUAL_ENV_NAME="DLP"

REM ���z����activate
@echo on
call activate %VIRTUAL_ENV_NAME%

call python deeplearning_for_photos.py