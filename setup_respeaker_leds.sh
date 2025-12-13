#!/bin/bash
# ReSpeaker LED Setup Script for Pascal AI Assistant
# Fixes USB permissions so LEDs work without sudo

echo "🔧 Setting up ReSpeaker LED permissions..."

# Create udev rule for ReSpeaker 4-Mic Array
UDEV_RULE='SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", MODE="0666", GROUP="plugdev"'

echo "Creating udev rule for ReSpeaker USB device..."
echo "$UDEV_RULE" | sudo tee /etc/udev/rules.d/99-respeaker.rules > /dev/null

# Reload udev rules
echo "Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

# Add user to plugdev group (if not already)
echo "Adding $USER to plugdev group..."
sudo usermod -a -G plugdev $USER

echo ""
echo "✅ Setup complete!"
echo ""
echo "⚠️  IMPORTANT: You need to either:"
echo "   1. Unplug and replug the ReSpeaker USB device, OR"
echo "   2. Reboot your Pi"
echo ""
echo "After that, run: python main.py --voice"
echo "The LEDs should now work without sudo!"
