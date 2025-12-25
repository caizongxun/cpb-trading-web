#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Upload models_v6 Folder to HuggingFace

高效批量上傳 models_v6 整個資料夾到 HuggingFace，避免 API 限制
一次性上傳整個資料夾，而非逐個上傳檔案

Usage in Colab:
    !pip install huggingface-hub -q
    !python upload_models_v6_batch.py

Or locally:
    python upload_models_v6_batch.py
"""

import os
import sys
from pathlib import Path
from typing import Optional
import time

try:
    from huggingface_hub import HfApi, Repository
except ImportError:
    print("Error: huggingface-hub 未安裝")
    print("請執行: pip install huggingface-hub")
    sys.exit(1)

# ============================================================================
# 配置
# ============================================================================

HF_REPO_ID = "zongowo111/cpb-models"  # HF 倉庫 ID
LOCAL_MODELS_DIR = "models_v6"         # 本地 models_v6 目錄
REPO_TYPE = "dataset"                  # 資料集類型
COMMIT_MESSAGE = "Batch upload V6 models from training"
MAX_RETRIES = 3                         # 最大重試次數
RETRY_DELAY = 5                         # 重試延遲 (秒)

print("\n" + "="*80)
print("HuggingFace Batch Uploader - models_v6 Directory")
print("="*80)

# ============================================================================
# 驗證與準備
# ============================================================================

class UploadValidator:
    """驗證上傳前的環境和資料夾"""
    
    @staticmethod
    def check_local_directory(dir_path: str) -> bool:
        """檢查本地 models_v6 目錄是否存在且非空"""
        print(f"\n[檢查] 本地目錄: {dir_path}")
        
        if not os.path.isdir(dir_path):
            print(f"  ✗ 目錄不存在")
            return False
        
        files = list(Path(dir_path).glob("*"))
        if not files:
            print(f"  ✗ 目錄為空")
            return False
        
        print(f"  ✓ 目錄存在，包含 {len(files)} 個檔案")
        
        # 統計檔案類型
        h5_count = len(list(Path(dir_path).glob("*.h5")))
        json_count = len(list(Path(dir_path).glob("*.json")))
        keras_count = len(list(Path(dir_path).glob("*.keras")))
        
        print(f"  - .h5 檔案: {h5_count}")
        print(f"  - .json 檔案: {json_count}")
        print(f"  - .keras 檔案: {keras_count}")
        
        total_size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"  - 總大小: {total_size_mb:.2f} MB")
        
        return True
    
    @staticmethod
    def check_hf_token() -> Optional[str]:
        """檢查 HF Token 是否可用"""
        print(f"\n[檢查] HuggingFace Token")
        
        # 優先檢查環境變數
        token = os.environ.get('HF_TOKEN')
        if token:
            print(f"  ✓ 從環境變數讀取 Token")
            return token
        
        # 檢查 Colab Secrets
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
            if token:
                print(f"  ✓ 從 Colab Secrets 讀取 Token")
                return token
        except:
            pass
        
        # 檢查本地 ~/.huggingface/token
        hf_token_path = Path.home() / ".huggingface" / "token"
        if hf_token_path.exists():
            token = hf_token_path.read_text().strip()
            print(f"  ✓ 從本地檔案讀取 Token")
            return token
        
        print(f"  ✗ 無法找到 HF Token")
        print(f"\n  方法 1: 設置環境變數")
        print(f"    export HF_TOKEN='your_token_here'")
        print(f"\n  方法 2: Colab Secrets (左側菜單)")
        print(f"    Key: HF_TOKEN, Value: 你的 token")
        print(f"\n  方法 3: 本地設置")
        print(f"    huggingface-cli login")
        
        return None
    
    @staticmethod
    def validate_repo(api: HfApi, repo_id: str, repo_type: str) -> bool:
        """驗證 HF 倉庫是否存在和可訪問"""
        print(f"\n[檢查] HuggingFace 倉庫: {repo_id}")
        
        try:
            info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
            print(f"  ✓ 倉庫存在且可訪問")
            print(f"  - ID: {info.repo_id}")
            print(f"  - 類型: {repo_type}")
            return True
        except Exception as e:
            print(f"  ✗ 無法訪問倉庫: {e}")
            return False

# ============================================================================
# 批量上傳
# ============================================================================

class BatchUploader:
    """高效批量上傳類"""
    
    def __init__(self, api: HfApi, repo_id: str, repo_type: str, token: str):
        self.api = api
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.token = token
        self.uploaded_count = 0
        self.failed_count = 0
        self.start_time = None
    
    def upload_directory(self, local_dir: str, remote_dir: str = None) -> bool:
        """
        一次性上傳整個目錄到 HF
        
        Args:
            local_dir: 本地目錄路徑
            remote_dir: HF 上的遠端目錄 (預設: 目錄名)
            
        Returns:
            bool: 是否上傳成功
        """
        self.start_time = time.time()
        
        if not remote_dir:
            remote_dir = Path(local_dir).name
        
        print(f"\n[開始] 批量上傳目錄")
        print(f"  本地: {local_dir}")
        print(f"  遠端: {self.repo_id}/{remote_dir}")
        print("-" * 80)
        
        # 收集所有檔案
        local_path = Path(local_dir)
        files = list(local_path.glob("*"))
        
        if not files:
            print("✗ 目錄為空，無法上傳")
            return False
        
        # 準備檔案列表
        file_list = []
        for file in sorted(files):
            if file.is_file():
                local_file = str(file)
                remote_file = f"{remote_dir}/{file.name}"
                file_list.append((local_file, remote_file))
        
        print(f"\n準備上傳 {len(file_list)} 個檔案")
        for local_file, remote_file in file_list:
            size_mb = os.path.getsize(local_file) / (1024 * 1024)
            print(f"  • {Path(local_file).name:30s} ({size_mb:6.2f} MB) → {remote_file}")
        
        # 執行上傳
        print("\n" + "-" * 80)
        print("[上傳中] 一次性上傳所有檔案...")
        
        try:
            # 使用 upload_folder 一次性上傳整個目錄
            # 這是最有效的方式，避免觸發 API 限制
            for attempt in range(MAX_RETRIES):
                try:
                    self.api.upload_folder(
                        folder_path=local_dir,
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                        path_in_repo=remote_dir,
                        token=self.token,
                        commit_message=COMMIT_MESSAGE,
                        multi_commit=False,  # 關鍵：一個 commit，而非多個
                        multi_commit_skip_errors=False,  # 有錯誤停止
                    )
                    
                    self.uploaded_count = len(file_list)
                    print(f"✓ 上傳成功！")
                    return True
                
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = RETRY_DELAY * (attempt + 1)
                        print(f"\n⚠ 第 {attempt + 1} 次嘗試失敗: {str(e)[:100]}")
                        print(f"等待 {wait_time} 秒後重試...")
                        time.sleep(wait_time)
                    else:
                        raise
        
        except Exception as e:
            print(f"✗ 上傳失敗 (全部重試已用盡)")
            print(f"錯誤: {e}")
            self.failed_count = len(file_list)
            return False
    
    def print_summary(self) -> None:
        """列印上傳統計"""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        print("\n" + "=" * 80)
        print("上傳統計")
        print("=" * 80)
        print(f"成功: {self.uploaded_count}")
        print(f"失敗: {self.failed_count}")
        print(f"耗時: {minutes}m {seconds}s")
        
        if self.uploaded_count > 0:
            print(f"\n✓ 上傳完成！")
            print(f"位置: https://huggingface.co/datasets/{self.repo_id}/tree/main/models_v6")
        else:
            print(f"\n✗ 上傳失敗")
        
        print("=" * 80)

# ============================================================================
# 主程式
# ============================================================================

def main():
    # 1. 驗證
    print("\n[步驟 1] 驗證環境與資料夾")
    print("-" * 80)
    
    if not UploadValidator.check_local_directory(LOCAL_MODELS_DIR):
        print("\n✗ 本地目錄檢查失敗")
        return False
    
    token = UploadValidator.check_hf_token()
    if not token:
        print("\n✗ 無法找到 HF Token")
        return False
    
    # 2. 初始化 API
    print("\n[步驟 2] 初始化 HuggingFace API")
    print("-" * 80)
    
    try:
        api = HfApi(token=token)
        print(f"✓ API 初始化成功")
    except Exception as e:
        print(f"✗ API 初始化失敗: {e}")
        return False
    
    # 3. 驗證倉庫
    print("\n[步驟 3] 驗證 HuggingFace 倉庫")
    print("-" * 80)
    
    if not UploadValidator.validate_repo(api, HF_REPO_ID, REPO_TYPE):
        print("\n✗ 倉庫驗證失敗")
        return False
    
    # 4. 執行上傳
    print("\n[步驟 4] 批量上傳")
    print("-" * 80)
    
    uploader = BatchUploader(api, HF_REPO_ID, REPO_TYPE, token)
    success = uploader.upload_directory(LOCAL_MODELS_DIR, "models_v6")
    
    # 5. 列印統計
    uploader.print_summary()
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ 使用者中斷")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ 發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
