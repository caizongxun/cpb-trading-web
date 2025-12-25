# Batch Upload models_v6 to HuggingFace

## Overview

After training is complete, you have a local `models_v6/` folder containing 60 model files.

**Key Point**: Upload the entire folder at once, not individual files, to avoid triggering HF API rate limits.

---

## Method 1: Using Batch Upload Script (Recommended)

### In Colab

After training completes:

```bash
!pip install huggingface-hub -q
!python upload_models_v6_batch.py
```

### On Local Machine

```bash
export HF_TOKEN='your_token_here'
python upload_models_v6_batch.py
```

### Upload Process

```
[Step 1] Validate Environment
  Check local models_v6 folder
  Check HF Token
  Validate HF repository access

[Step 2] Initialize HF API
  Connect to HuggingFace
  Verify repository exists

[Step 3] Batch Upload
  Upload entire models_v6 folder in one batch
  Auto-retry mechanism (3 attempts)
  Progress monitoring

[Step 4] Completion Verification
  Display upload statistics
  Provide HF repository link
```

---

## Method 2: Python API (Manual)

### Simple One-Click Upload

```python
from huggingface_hub import HfApi

HF_TOKEN = "your_hf_token_here"
HF_REPO_ID = "zongowo111/cpb-models"
LOCAL_DIR = "models_v6"
REMOTE_DIR = "models_v6"

api = HfApi(token=HF_TOKEN)

print("Uploading models_v6 folder...")

api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    path_in_repo=REMOTE_DIR,
    token=HF_TOKEN,
    commit_message="Batch upload V6 models",
    multi_commit=False,  # Key: single commit, not multiple
)

print("Upload complete!")
print(f"Location: https://huggingface.co/datasets/{HF_REPO_ID}/tree/main/{REMOTE_DIR}")
```

---

## Method 3: HF CLI (Terminal)

### Login First

```bash
huggingface-cli login
# Enter your HF token
```

### Upload Folder

```bash
huggingface-cli upload \
  zongowo111/cpb-models \
  models_v6 \
  --repo-type dataset \
  --commit-message "Batch upload V6 models"
```

---

## upload_models_v6_batch.py Details

### Features

✅ **Batch Upload** - Upload entire folder in one commit (avoids API limits)
✅ **Auto-Retry** - 3 retry attempts with exponential backoff (5s, 10s, 15s)
✅ **Environment Validation** - Check local folder, HF token, repository access
✅ **Progress Monitoring** - Detailed upload logs and statistics
✅ **Multiple Token Sources** - Environment variable, Colab Secrets, local file
✅ **Error Handling** - Comprehensive exception handling and recovery

### Configuration

```python
HF_REPO_ID = "zongowo111/cpb-models"  # HF repository ID
LOCAL_MODELS_DIR = "models_v6"         # Local folder name
REPO_TYPE = "dataset"                  # Repository type
COMMIT_MESSAGE = "Batch upload V6 models from training"
MAX_RETRIES = 3                         # Max retry attempts
RETRY_DELAY = 5                         # Initial retry delay (seconds)
```

---

## Complete Colab Workflow

```bash
# [Step 1] Navigate to project (training already complete)
%cd cpb-trading-web

# [Step 2] Install dependencies
!pip install huggingface-hub -q

# [Step 3] Verify models_v6 folder exists
!ls -lh models_v6/ | head -20
# Should show 60 .h5 files and 60 .json files

# [Step 4] Ensure HF Token is set in Secrets
# Left sidebar → Secrets → Add new secret
# Key: HF_TOKEN
# Value: your HuggingFace token

# [Step 5] Execute batch upload
!python upload_models_v6_batch.py
```

### Expected Output

```
================================================================================
HuggingFace Batch Uploader - models_v6 Directory
================================================================================

[Check] Local Directory: models_v6
  ✓ Directory exists, contains 120 files
  - .h5 files: 60
  - .json files: 60
  - Total size: 3012.45 MB

[Check] HuggingFace Token
  ✓ Token loaded from Colab Secrets

[Check] HuggingFace Repository: zongowo111/cpb-models
  ✓ Repository accessible
  - Type: dataset

[Begin] Batch Upload
  Local: models_v6
  Remote: zongowo111/cpb-models/models_v6

Preparing 120 files...
  • BTC_1d.h5 (50.23 MB)
  • BTC_1d_metrics.json (0.50 KB)
  • BTC_1h.h5 (48.90 MB)
  ...
  • HBAR_15m_metrics.json (0.50 KB)

[Upload] Batch uploading all files...
✓ Upload successful!

================================================================================
Upload Summary
================================================================================
Success: 120
Failed: 0
Elapsed Time: 12m 34s

✓ Upload Complete!
Location: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/models_v6
================================================================================
```

---

## FAQ

### Q: Why upload the entire folder at once?

A: To avoid triggering HF API rate limits. Uploading 120 files individually would create 120 API requests, which could hit rate limits. Batch uploading is one operation.

### Q: How long does upload take?

A: Depends on internet speed:
- Fast internet (100+ Mbps): 5-10 minutes
- Standard broadband (50 Mbps): 10-20 minutes
- Slower connection (10 Mbps): 20-30 minutes

### Q: What if upload is interrupted?

A:
1. Script has auto-retry mechanism (3 attempts)
2. If all fail, simply re-run: `!python upload_models_v6_batch.py`
3. Or use HF CLI: `huggingface-cli upload ...`

### Q: How to verify successful upload?

A:
1. Visit https://huggingface.co/datasets/zongowo111/cpb-models
2. Click "Files and versions" tab
3. Check `models_v6/` folder for 120 files

### Q: Token expired?

A:
1. Generate new token: https://huggingface.co/settings/tokens
2. Update Colab Secrets or environment variable
3. Re-run upload script

---

## Best Practices

### Before Upload

```bash
# Verify file count
ls models_v6/ | wc -l
# Should show: 120

# Verify total size
du -sh models_v6/
# Should show: ~3 GB

# Verify file types
ls models_v6/*.h5 | wc -l
# Should show: 60
ls models_v6/*.json | wc -l
# Should show: 60
```

### Before Uploading

1. Verify HF token is valid and has write permissions
2. Verify local `models_v6/` folder exists and is not empty
3. Verify stable internet connection
4. Allow sufficient time (10-30 minutes)

### After Upload

```bash
# Visit HF repository
https://huggingface.co/datasets/zongowo111/cpb-models

# Check models_v6 folder
https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/models_v6

# Should see 120 files total (60 .h5 + 60 .json)
```

---

## Summary

**Use the batch upload script**:
```bash
!python upload_models_v6_batch.py
```

This is the simplest, safest, and most efficient way to upload the entire `models_v6` folder to HuggingFace, avoiding API rate limits and providing complete progress monitoring and error recovery.

**File location**: `cpb-trading-web/upload_models_v6_batch.py`
