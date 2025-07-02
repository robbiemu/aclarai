#!/usr/bin/env python3
"""
Test script to launch the Gradio import panel and verify the UI.
This allows visual verification of the import panel functionality.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the UI module to path  
sys.path.insert(0, 'services/aclarai-ui')

def main():
    """Launch the Gradio import panel for testing."""
    print("🚀 Launching Gradio Import Panel Test")
    print("=" * 50)
    
    try:
        from aclarai_ui.main import create_import_interface
        
        print("✅ Creating Gradio interface...")
        interface = create_import_interface()
        
        print("🌐 Launching on localhost:7860...")
        print("📝 You can test the following scenarios:")
        print("   1. Upload a .txt file with conversation format")
        print("   2. Upload a .json file to test fallback")
        print("   3. Upload an empty file to test error handling")
        print("   4. Clear the queue to test queue management")
        print("\n🔍 Expected behavior:")
        print("   - Files with conversation patterns: ⚠️ Fallback (DefaultPlugin)")
        print("   - Files without conversations: ⏸️ Skipped")
        print("   - Real-time queue updates and summary statistics")
        print("\n💡 Press Ctrl+C to stop the server")
        
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\n✅ Server stopped by user")
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())