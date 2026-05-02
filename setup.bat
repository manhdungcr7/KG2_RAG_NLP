@echo off
echo ===================================================
echo CHUONG TRINH CAI DAT MOI TRUONG CHO KG2RAG (WINDOWS)
echo ===================================================
echo.

echo [*] Buoc 1: Kiem tra va tao moi truong ao (venv)...
IF NOT EXIST "venv" (
    python -m venv venv
    echo [OK] Da tao thanh cong thu muc venv.
) ELSE (
    echo [!] Thu muc venv da ton tai, bo qua buoc tao.
)
echo.

echo [*] Buoc 2: Kich hoat moi truong ao va cai dat thu vien...
call venv\Scripts\activate.bat
echo Dang cap nhat pip va doc file requirements.txt...
pip install --upgrade pip
pip install -r requirements.txt
echo.

echo ===================================================
echo [HOAN TAT] Moi truong da duoc cai dat thanh cong!
echo ===================================================
echo.
echo [!] LUU Y QUAN TRONG TRUOC KHI CHAY CODE:
echo 1. Data KHONG duoc dua len GitHub. Hay xin link Google Drive de tai thu muc "data/" va dat vao day.
echo 2. De bat dau lam viec, hay mo Terminal (PowerShell) tai thu muc nay va go lenh:
echo    .\venv\Scripts\activate
echo.
pause