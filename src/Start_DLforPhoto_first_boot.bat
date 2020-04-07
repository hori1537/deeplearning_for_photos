@echo off
echo プログラム開始

REM start_programのあるフォルダに戻る
cd /d %~dp0

REM condaが実行できるか確認

call conda -V

if %errorlevel% neq 0 ( 
echo condaのPATHが通っていません
echo Anaconda プロンプトを起動し、start.batを実行してください
echo 10秒後にプログラムを終了します
TIMEOUT /T 10
exit /B
) else (
echo condaコマンドは正常に実行されました
)

SET VIRTUAL_ENV_NAME="DLP"

REM 仮想環境をactivate
@echo on
call activate %VIRTUAL_ENV_NAME%
@echo off

if %errorlevel% neq 0 (
echo 仮想環境のactivate失敗
echo 仮想環境の作成開始
@echo on
echo Y | call conda create -n %VIRTUAL_ENV_NAME% python=3.6
call activate %VIRTUAL_ENV_NAME%
) else (
@echo off
echo 仮想環境をactivateしました
)

@echo on
REM 必要なライブラリをpip install
call pip install -r requirements.txt
call python deeplearning_for_photos.py