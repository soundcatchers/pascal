# 🔄 Vosk Model Upgrade Guide (0.15 → 0.22)

Upgrade your Pascal voice recognition from **vosk-model-small-en-us-0.15** (75% accuracy) to **vosk-model-en-us-0.22** (85% accuracy) for significantly better speech recognition.

---

## 📊 What's New in 0.22

| Feature | 0.15 (Old) | **0.22 (New)** | Improvement |
|---------|------------|----------------|-------------|
| **Accuracy** | ~75% | **~85%** | +20% better |
| **Model Size** | 50MB | 1.8GB | Larger but more accurate |
| **RAM Usage** | ~500MB | ~2GB | Pi 5 handles easily (16GB RAM) |
| **Speed** | Real-time | Real-time | Same |
| **Offline** | ✅ Yes | ✅ Yes | Same |
| **Cost** | FREE | FREE | Same |

**Common improvements:**
- "whims it built" → "when was it built" ✅
- "Brighten" → "Brighton" ✅
- "quit" → "quit" (better exit command recognition) ✅

---

## 🚀 Quick Upgrade (Recommended)

### Option 1: Automatic Upgrade (Easiest)

```bash
# 1. Remove old model
rm -rf config/vosk_models/vosk-model-small-en-us-0.15

# 2. Download new model (automatic)
./setup_vosk.sh

# 3. Restart Pascal
./run.sh --voice
```

**Done!** Pascal will automatically detect and use the new 0.22 model.

---

### Option 2: Manual Upgrade

```bash
# 1. Remove old model
rm -rf config/vosk_models/vosk-model-small-en-us-0.15

# 2. Download new model manually
cd config/vosk_models
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
rm vosk-model-en-us-0.22.zip
cd ../..

# 3. Verify installation
ls -lh config/vosk_models/

# Expected output:
# vosk-model-en-us-0.22/  (1.8GB)

# 4. Restart Pascal
./run.sh --voice
```

---

## ⚙️ Compatibility

### Backward Compatibility
Pascal automatically detects which model is installed:
1. **Prefers 0.22** if available (best accuracy)
2. **Falls back to 0.15** if 0.22 not found (backward compatible)

You can keep both models installed, and Pascal will use 0.22 automatically.

### Storage Requirements
- **Before upgrade**: 50MB used
- **After upgrade**: 1.8GB used
- **Difference**: +1.75GB

**Pi 5 users**: With a 32GB+ SD card, this is negligible.

---

## ✅ Verify Upgrade

After upgrading, run Pascal in voice mode:

```bash
./run.sh --voice --debug-audio
```

**Look for this line in the startup logs:**
```
[STT] Loading Vosk model from: config/vosk_models/vosk-model-en-us-0.22
[STT] ✅ Vosk initialized (sample rate: 16000Hz)
```

**If you see:**
```
[STT] Loading Vosk model from: config/vosk_models/vosk-model-small-en-us-0.15
```
Then the old model is still being used. Make sure you deleted it and re-ran `./setup_vosk.sh`.

---

## 🧪 Test the Improvement

Try these test phrases to compare accuracy:

**Before (0.15)**:
- "when was it built" → "whims it built" ❌
- "Brighton" → "Brighten" ❌
- "quit" → "quick" or "quite" ❌

**After (0.22)**:
- "when was it built" → "when was it built" ✅
- "Brighton" → "Brighton" ✅
- "quit" → "quit" ✅

---

## ⚠️ Troubleshooting

### "Model not found" after upgrade
```bash
# Verify the new model exists
ls -la config/vosk_models/

# Expected:
# vosk-model-en-us-0.22/

# If missing, re-download:
./setup_vosk.sh
```

### Still using old model (0.15)
```bash
# Delete old model completely
rm -rf config/vosk_models/vosk-model-small-en-us-0.15

# Restart Pascal
./run.sh --voice
```

### Download fails or is slow
The 0.22 model is 1.8GB. On slower connections:
- **10 Mbps**: ~24 minutes
- **50 Mbps**: ~5 minutes
- **100 Mbps**: ~2 minutes

If download fails, try manual download with resume support:
```bash
cd config/vosk_models
wget -c https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
```

### Not enough storage space
If you're low on storage (<2GB free), stick with 0.15 model:
```bash
# Keep using 0.15 (no upgrade needed)
# Pascal will continue working with the old model
```

---

## 🔄 Rollback to 0.15

If you want to go back to the smaller model:

```bash
# 1. Remove 0.22 model
rm -rf config/vosk_models/vosk-model-en-us-0.22

# 2. Download 0.15 model
cd config/vosk_models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
rm vosk-model-small-en-us-0.15.zip
cd ../..

# 3. Restart Pascal
./run.sh --voice
```

---

## 📈 Performance Impact

**RAM Usage:**
- Before: ~500MB
- After: ~2GB
- **Impact on Pi 5**: Negligible (you have 16GB total)

**CPU Usage:**
- Same as before (real-time transcription)

**Recognition Speed:**
- Same as before (<200ms latency)

**Accuracy:**
- **+20% improvement** in word error rate (WER)

---

## 🎯 Summary

| Action | Command |
|--------|---------|
| **Upgrade** | `rm -rf config/vosk_models/vosk-model-small-en-us-0.15 && ./setup_vosk.sh` |
| **Verify** | `./run.sh --voice --debug-audio` |
| **Rollback** | See "Rollback to 0.15" section above |

**Recommended**: Upgrade to 0.22 for significantly better voice recognition accuracy! 🎉
