"""
ReSpeaker LED Controller - Visual feedback for Pascal AI Assistant
Controls the 12 RGB LEDs on ReSpeaker 4-Mic Array USB

States:
- idle: LEDs off
- listening: Blue spinning animation
- thinking: Yellow/orange pulse
- speaking: Green glow
- error: Red flash
- wakeup: Quick blue pulse
- shutdown: Fade to off
"""

import time
import threading
from typing import Optional

try:
    from pixel_ring import pixel_ring
    PIXEL_RING_AVAILABLE = True
except ImportError:
    PIXEL_RING_AVAILABLE = False
    print("[LED] ⚠️  pixel_ring not installed. Install with: pip install pixel-ring pyusb")

try:
    import usb.core
    USB_AVAILABLE = True
except ImportError:
    USB_AVAILABLE = False


class LEDController:
    """Controls ReSpeaker LED ring for visual feedback"""
    
    RESPEAKER_VENDOR_ID = 0x2886
    RESPEAKER_PRODUCT_ID = 0x0018
    
    def __init__(self, enabled: bool = True, brightness: int = 50):
        self.enabled = enabled and PIXEL_RING_AVAILABLE
        self.brightness = brightness
        self.current_state = "idle"
        self.available = False
        self._lock = threading.Lock()
        
        if self.enabled:
            self._initialize()
    
    def _initialize(self) -> bool:
        """Initialize LED controller"""
        if not PIXEL_RING_AVAILABLE:
            print("[LED] ❌ pixel_ring library not available")
            return False
        
        try:
            if USB_AVAILABLE:
                dev = usb.core.find(idVendor=self.RESPEAKER_VENDOR_ID, 
                                   idProduct=self.RESPEAKER_PRODUCT_ID)
                if not dev:
                    print("[LED] ⚠️  ReSpeaker USB device not found (LEDs disabled)")
                    self.enabled = False
                    return False
            
            pixel_ring.set_brightness(self.brightness)
            pixel_ring.off()
            self.available = True
            print(f"[LED] ✅ ReSpeaker LEDs initialized (brightness: {self.brightness}%)")
            return True
            
        except Exception as e:
            print(f"[LED] ⚠️  LED initialization failed: {e}")
            self.enabled = False
            return False
    
    def _safe_call(self, func, *args, **kwargs):
        """Thread-safe LED function call with error handling"""
        if not self.enabled or not self.available:
            return
        
        with self._lock:
            try:
                func(*args, **kwargs)
            except Exception as e:
                pass
    
    def idle(self):
        """LEDs off - idle/standby state"""
        self.current_state = "idle"
        self._safe_call(pixel_ring.off)
    
    def listening(self):
        """Blue spinning animation - actively listening"""
        self.current_state = "listening"
        self._safe_call(pixel_ring.listen)
    
    def thinking(self):
        """Yellow/orange pulse - processing query"""
        self.current_state = "thinking"
        self._safe_call(pixel_ring.think)
    
    def speaking(self):
        """Green glow - AI responding"""
        self.current_state = "speaking"
        self._safe_call(pixel_ring.speak)
    
    def wakeup(self):
        """Quick blue pulse - wake word detected"""
        self.current_state = "wakeup"
        self._safe_call(pixel_ring.wakeup)
    
    def error(self):
        """Red flash - error occurred"""
        self.current_state = "error"
        self._safe_call(pixel_ring.mono, 255, 0, 0)
    
    def success(self):
        """Green flash - operation successful"""
        self.current_state = "success"
        self._safe_call(pixel_ring.mono, 0, 255, 0)
    
    def custom_color(self, r: int, g: int, b: int):
        """Set custom RGB color (0-255 each)"""
        self.current_state = "custom"
        self._safe_call(pixel_ring.mono, r, g, b)
    
    def set_brightness(self, level: int):
        """Set LED brightness (0-100)"""
        self.brightness = max(0, min(100, level))
        self._safe_call(pixel_ring.set_brightness, self.brightness)
    
    def shutdown(self):
        """Graceful shutdown - brief flash then off"""
        if not self.enabled or not self.available:
            return
        
        try:
            with self._lock:
                pixel_ring.mono(100, 100, 255)
                time.sleep(0.3)
                pixel_ring.off()
            self.current_state = "off"
            print("[LED] ✅ LEDs shut down")
        except Exception:
            pass
    
    def off(self):
        """Turn off LEDs immediately"""
        self._safe_call(pixel_ring.off)
        self.current_state = "off"
    
    def get_state(self) -> str:
        """Get current LED state"""
        return self.current_state
    
    def is_available(self) -> bool:
        """Check if LEDs are available"""
        return self.enabled and self.available


_led_controller: Optional[LEDController] = None


def get_led_controller(enabled: bool = True, brightness: int = 50) -> LEDController:
    """Get or create the global LED controller instance"""
    global _led_controller
    if _led_controller is None:
        _led_controller = LEDController(enabled=enabled, brightness=brightness)
    return _led_controller
