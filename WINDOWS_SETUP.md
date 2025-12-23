# Windows PowerShell 設置指南

## 第一步：下載這個存倉库

```powershell
# 儙待 Desktop 走進
cd Desktop

# 下載存倉库
git clone https://github.com/caizongxun/cpb-trading-web.git
cd cpb-trading-web

# 確認是否成功進入目錄
dir
```

## 第二步：建立 `.env` 檔案

在 PowerShell 中：

```powershell
# 複製模版
(Get-Content .env.example) | Set-Content .env

# 置陳 .env 檔案 (用 Notepad)
notepads .env
```

編輯 `.env` 檔案，物錄目下這些位置：

```
HF_TOKEN=hf_你的你的token_例如hf_ABC123XYZ
HF_USERNAME=zongowo111
```

儲存（Ctrl+S）並钱上 Notepad

## 第三步：設置環境變數

在 PowerShell 中：

```powershell
# Windows PowerShell 設置環境變數的方式 (會推謜爲方法)
$env:HF_TOKEN="你的token"
$env:HF_USERNAME="zongowo111"

# 驗證是否設置成功
echo $env:HF_TOKEN
echo $env:HF_USERNAME
```

## 第四步：安裝 Python 依賴

在 PowerShell 中（確保是在 `cpb-trading-web` 目錄）：

```powershell
# 先殾綜一下目錄
ls

# 安裝依賴
pip install -r requirements.txt

# 耙個操作會需要幾分鐘...
```

如果出現任何错誤，試試：

```powershell
# 升級 pip
python -m pip install --upgrade pip

# 重新安裝
pip install -r requirements.txt --upgrade
```

## 第五步：運行後端 FastAPI

在 PowerShell 中：

```powershell
# 運行後端
python app.py

# 或者，使用 uvicorn 的海淊模式（修改代碼會自動重載）
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

看到這個載線時表示成功：

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

聴不要關這個 PowerShell 窗下。

## 第六步：(另一個 PowerShell) 運行前端

打開一個**新的** PowerShell 窗下：

```powershell
# 走進目錄
cd C:\Users\zong\Desktop\cpb-trading-web

# 訟済 HTTP 伺务器
python -m http.server 5000

# 看到這些載線：
Servving HTTP on 0.0.0.0 port 5000
```

聴不要關這個 PowerShell 窗下。

## 第七步：打開前端網頁

用瀏覽器打開：

```
http://localhost:5000
```

就會看到美麗的絥沀預測頁面了！

---

## 整理、其實你只需要 3 個 PowerShell 窗下

### 窗下 1: 第一鍵設置 + 安裝

```powershell
cd Desktop
git clone https://github.com/caizongxun/cpb-trading-web.git
cd cpb-trading-web
(Get-Content .env.example) | Set-Content .env
# 修改 .env (你的 token)
pip install -r requirements.txt
```

### 窗下 2: 運行後端

```powershell
cd C:\Users\zong\Desktop\cpb-trading-web
$env:HF_TOKEN="你的token"
$env:HF_USERNAME="zongowo111"
python app.py
```

### 窗下 3: 運行前端

```powershell
cd C:\Users\zong\Desktop\cpb-trading-web
python -m http.server 5000
```

### 窗下 4: 打開海覽器

```
http://localhost:5000
```

---

## 常見故障

### 1. ModuleNotFoundError

```
ModuleNotFoundError: No module named 'fastapi'
```

解決：確保已執行 `pip install -r requirements.txt`

### 2. 應用程式不是内部或外部下令

```powershell
export is not recognized as an internal or external command
```

解決：在 Windows PowerShell 中，使用 `$env:` 而不是 `export`

```powershell
# 錯佬
# export HF_TOKEN="..."

# 正確
$env:HF_TOKEN="..."
```

### 3. 找不到檔案

```
PathNotFound
```

解決：确定你是位了正確的目錄

```powershell
# 確認位置
cd C:\Users\zong\Desktop\cpb-trading-web
dir  # 應該看到 app.py, index.html, requirements.txt
```

### 4. 端口 8000/5000 已被佔用

```powershell
# 改一个端口，例如 8001
uvicorn app:app --host 0.0.0.0 --port 8001

# 或前端的 5001
python -m http.server 5001
```

---

## 三個窗下都被走了？

窗下 1 的推薦信號 → 窗下 2 是後端 → 窗下 3 是前端（HTTP 伺务器）

了解後，剪一下這個指南並保存。
