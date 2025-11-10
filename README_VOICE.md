# 🎤 Pascal Voice Input Setup Guide

Complete setup instructions for offline voice input on Raspberry Pi 5 with ReSpeaker USB Mic Array.

---

## 📋 Overview

Pascal now supports **offline, real-time voice input** using:
- **Vosk** - Fast, lightweight speech-to-text engine
- **ReSpeaker USB Mic Array** - Professional 4-mic array with built-in noise suppression
- **PyAudio** - Audio capture library

**Features:**
- ⚡ Real-time transcription (near-zero latency)
- 💰 100% offline, no API costs
- 🎙️  ReSpeaker auto-detection
- 🔊 Voice activity detection
- 💬 Hybrid mode (voice + text input)
- 🪶 Lightweight (~50MB model, <500MB RAM)

---

## 🛠️ Installation (Raspberry Pi 5)

### Step 1: Install System Dependencies

```bash
# Update system
sudo apt-get update

# Install PortAudio for PyAudio
sudo apt-get install -y portaudio19-dev python3-pyaudio

# Install USB support for ReSpeaker LED control (optional)
sudo apt-get install -y python3-pyusb
```

### Step 2: Install Python Packages

```bash
# Install Vosk and PyAudio
pip install vosk==0.3.45
pip install PyAudio==0.2.14

# Optional: ReSpeaker LED control
pip install pixel-ring==0.1.1
```

### Step 3: Download Vosk Model

```bash
# Run the setup script (easiest method)
./setup_vosk.sh

# Or manually:
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
mkdir -p config/vosk_models
mv vosk-model-small-en-us-0.15 config/vosk_models/
rm vosk-model-small-en-us-0.15.zip
```

### Step 4: Connect ReSpeaker Mic

```bash
# Plug in ReSpeaker USB Mic Array
# Should auto-detect on Raspberry Pi 5 (no drivers needed)

# Verify detection
python main.py --list-devices

# Expected output:
# ✅ ReSpeaker [2] ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:1,0)
#     Channels: 6, Sample Rate: 16000Hz
```

---

## 🎙️ Usage

### Voice-Only Mode (Always Listening)

```bash
# Recommended: Use run.sh (handles virtual environment)
./run.sh --voice

# Or directly:
python main.py --voice
```

**Features:**
- Continuous speech recognition
- Speak naturally, no wake word needed
- Say "exit" or "quit" to stop
- Can still type text commands if needed

### List Audio Devices

```bash
# Recommended: Use run.sh
./run.sh --list-devices

# Or directly:
python main.py --list-devices
```

Shows all available microphones with ReSpeaker highlighting.

### Normal Text Mode (Default)

```bash
# Recommended: Use run.sh
./run.sh

# Or directly:
python main.py
```

Voice input disabled, text-only interaction.

---

## 💬 Voice Interaction Examples

### Starting Pascal with Voice

```bash
$ ./run.sh --voice

🤖 Starting Pascal AI Assistant...
==================================
[INFO] Checking voice input dependencies...
[SUCCESS] Voice input dependencies OK
[INFO] Starting Pascal...

⚡ Pascal AI Assistant v4.1 - Enhanced Conversational Edition + Voice Input
🎙️  ReSpeaker Voice input ready: ReSpeaker 4 Mic Array (UAC1.0)
⚡ Pascal enhanced conversational system ready!

💬 Chat with Pascal Enhanced Conversational Edition (Voice + Text Mode)
🎙️  Voice mode active - speak naturally or type text

🎤 Listening: hello pascal...
🎤 You (voice): hello pascal

Pascal: Hey! I'm Pascal. What's up?
```

### Voice + Personality Switching

```bash
🎤 You (voice): Rick are you out there

Rick: Yeah yeah, Rick here. *burp* What do you want?

🎤 You (voice): what's the weather like

Rick: Oh geez, 18 degrees and partly cloudy. *burp* This is like, basic meteorology...
```

### Hybrid Mode (Voice + Text)

You can mix voice and text input freely:

```bash
🎤 You (voice): who won the last f1 race

Pascal: Lando Norris won the most recent Formula 1 race at...

You (typing): who came second

Pascal: Oscar Piastri finished in second place...
```

---

## 🔧 Troubleshooting

### Problem: "Vosk model not found"

**Solution:**
```bash
./setup_vosk.sh
```

Or check the model is in the correct location:
```bash
ls -la config/vosk_models/vosk-model-small-en-us-0.15/
```

### Problem: "No audio input device found"

**Solution:**
```bash
# List USB devices
lsusb | grep ReSpeaker

# Check ALSA devices
arecord -l

# Test microphone
arecord -d 5 -f cd test.wav
aplay test.wav
```

### Problem: PyAudio installation fails

**Solution:**
```bash
# Install PortAudio development headers first
sudo apt-get install portaudio19-dev

# Then install PyAudio
pip install PyAudio
```

### Problem: "Permission denied" for audio device

**Solution:**
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Log out and log back in for changes to take effect
```

### Problem: Poor recognition accuracy

**Solutions:**
1. **Reduce background noise** - ReSpeaker has built-in noise suppression but works best in quiet environments
2. **Speak clearly** - Enunciate words, avoid mumbling
3. **Check mic position** - ReSpeaker should be within 1-2 meters
4. **Use larger model** (requires more RAM):
   ```bash
   # Download medium model (~1GB)
   wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
   unzip vosk-model-en-us-0.22.zip
   mv vosk-model-en-us-0.22 config/vosk_models/
   ```

---

## ⚙️ Advanced Configuration

### ReSpeaker LED Control (Optional)

Control the LED ring on the ReSpeaker for visual feedback:

```python
from pixel_ring import pixel_ring

# Listening state
pixel_ring.listen()

# Thinking state
pixel_ring.think()

# Speaking state
pixel_ring.speak()

# Turn off
pixel_ring.off()
```

### Adjust Recognition Sensitivity

Edit `modules/speech_input.py`:

```python
# For better accuracy (slower):
self.recognizer.SetWords(True)  # Enable word-level timestamps

# For faster recognition (less accurate):
self.recognizer.SetWords(False)
```

### Custom Vosk Model

To use a different Vosk model:

```bash
# Download your preferred model
wget https://alphacephei.com/vosk/models/YOUR-MODEL.zip
unzip YOUR-MODEL.zip
mv YOUR-MODEL config/vosk_models/

# Update model path in speech_input.py or pass via environment:
export VOSK_MODEL_PATH="config/vosk_models/YOUR-MODEL"
```

---

## 📊 Performance on Raspberry Pi 5

| Metric | Value |
|--------|-------|
| **Latency** | ~100-200ms (near real-time) |
| **CPU Usage** | ~15-25% (single core) |
| **RAM Usage** | ~300-400MB |
| **Model Size** | 50MB (small model) |
| **Accuracy** | 85-95% (clear speech, quiet environment) |

**Benchmark:**
- Vosk small model on Pi 5: ~1.2x real-time processing
- 60 seconds of audio processed in ~60-75 seconds
- Streaming mode: Instant partial results

---

## 🎯 Tips for Best Performance

1. **Use ReSpeaker's built-in DSP**
   - Automatic noise suppression
   - Beamforming (focuses on speaker direction)
   - Echo cancellation

2. **Optimize environment**
   - Quiet room (< 50dB background noise)
   - 1-2 meters from microphone
   - Avoid loud music or TV in background

3. **Speak naturally**
   - Normal pace, don't rush
   - Clear enunciation
   - Avoid filler words ("um", "uh")

4. **Use text fallback**
   - Complex technical terms
   - Names of people/places
   - Long numbers or codes

5. **Monitor CPU temperature**
   ```bash
   # Check CPU temp
   vcgencmd measure_temp
   
   # If > 70°C, consider cooling
   ```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| **Vosk Models** | https://alphacephei.com/vosk/models |
| **Vosk Documentation** | https://alphacep.com/vosk/ |
| **ReSpeaker Wiki** | https://wiki.seeedstudio.com/ReSpeaker-USB-Mic-Array/ |
| **PyAudio Docs** | https://people.csail.mit.edu/hubert/pyaudio/docs/ |

---

## ❓ FAQ

**Q: Can I use a different microphone?**  
A: Yes! Any USB microphone works. ReSpeaker is recommended for best quality.

**Q: Does this work on Replit?**  
A: No, voice input requires local hardware (Pi 5 + microphone).

**Q: Can I use online STT instead?**  
A: Vosk is offline by design for privacy and zero cost. For online STT, consider Google/Azure APIs.

**Q: What languages are supported?**  
A: Vosk supports 20+ languages. Download the appropriate model from https://alphacephei.com/vosk/models

**Q: Can I use wake word detection?**  
A: Not yet. Wake word support (Mode 3) is planned for future release.

---

## 🚀 Next Steps

After getting voice input working:

1. **Test with different personalities:**
   ```bash
   ./run.sh --voice
   # Say: "Jarvis are you out there"
   # Say: "Rick are you out there"
   ```

2. **Experiment with follow-up questions:**
   ```bash
   # Say: "Who won the last F1 race?"
   # Say: "Where are they racing this weekend?"
   ```

3. **Try hybrid mode:**
   - Use voice for quick questions
   - Use text for complex commands or debugging

---

**Enjoy hands-free Pascal! 🎉**
