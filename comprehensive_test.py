#!/usr/bin/env python3
"""
Pascal AI Assistant - Comprehensive System Test
Tests all fixes and validates current info routing works correctly
"""

import asyncio
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Enable debug mode for comprehensive testing
os.environ['DEBUG'] = 'true'

async def test_comprehensive_system():
    """Run comprehensive system test with all fixes"""
    print("🔬 Pascal AI Assistant - Comprehensive System Test")
    print("=" * 60)
    
    try:
        # Test 1: Import all modules
        print("\n📦 Testing Module Imports:")
        print("-" * 40)
        
        from config.settings import settings
        from modules.router import LightningRouter
        from modules.personality import PersonalityManager
        from
