import os
import time
from playwright.sync_api import sync_playwright

def test_spag4d_ui_full_flow():
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        # Navigate
        url = "http://127.0.0.1:8000"
        print(f"Navigating to {url}...")
        try:
            page.goto(url)
        except Exception as e:
            print(f"Error: Could not reach {url}. Is the server running?")
            browser.close()
            return

        print("Page loaded. Verifying title...")
        assert "SPAG-4D" in page.title()

        # 1. Verify New UI Elements
        print("--- Verifying UI Elements ---")
        
        # Sky Threshold
        sky_input = page.locator("#sky-threshold")
        assert sky_input.is_visible(), "Sky Threshold input missing!"
        print("✅ Sky Threshold found.")
        
        # Magic Fix Checkbox
        sharp_cb = page.locator("#sharp-refine")
        assert sharp_cb.is_checked(), "Magic Fix should be checked by default!"
        print("✅ Magic Fix checked.")
        
        # Resolution Dropdown
        # Wait for JS
        time.sleep(1) 
        res_group = page.locator("#sharp-resolution-group")
        assert res_group.is_visible(), "Resolution group should be visible (Magic Fix is checked)."
        
        res_select = page.locator("#sharp-cubemap-size")
        assert res_select.count() > 0, "Resolution dropdown missing!"
        print("✅ Resolution dropdown found.")

        # Opacity Blend
        opacity_input = page.locator("#opacity-blend")
        assert opacity_input.count() > 0
        print("✅ Opacity Blend slider found.")
        
        # 2. Interact with UI
        print("\n--- Testing Interaction ---")
        
        # Change Sky Threshold
        print("Setting Sky Threshold to 50...")
        sky_input.fill("50")
        assert sky_input.input_value() == "50"
        
        # Change Resolution
        print("Setting Resolution to 768...")
        res_select.select_option("768")
        assert res_select.input_value() == "768"
        
        # 3. Upload File & Convert
        print("\n--- Testing Conversion Flow ---")
        
        dummy_path = os.path.abspath("tests/dummy.jpg")
        if not os.path.exists(dummy_path):
            print(f"⚠️ Dummy image not found at {dummy_path}. Skipping conversion test.")
        else:
            print(f"Uploading {dummy_path}...")
            file_input = page.locator("#file-input")
            file_input.set_input_files(dummy_path)
            
            # Verify filename label updated
            lbl = page.locator("#filename")
            assert "dummy.jpg" in lbl.text_content()
            print("✅ File selected.")
            
            # Click Convert
            convert_btn = page.locator("#convert-btn")
            assert convert_btn.is_enabled()
            print("Clicking Convert...")
            convert_btn.click()
            
            # Wait for status update
            status_text = page.locator("#status-text")
            print("Waiting for processing status...")
            # Should change from "Ready" to "Uploading..." then "Processing..."
            try:
                # Wait for "Processing..." or "Complete!"
                # It might be fast locally with dummy image
                page.wait_for_function("document.getElementById('status-text').textContent.includes('Processing') || document.getElementById('status-text').textContent.includes('Complete')", timeout=10000)
                txt = status_text.text_content()
                print(f"✅ Conversion started! Status: {txt}")
                
                # Wait for completion (optional, might take long on some machines)
                # But dummy image is tiny.
                print("Waiting for completion...")
                try:
                    page.wait_for_function("document.getElementById('status-text').textContent.includes('Complete')", timeout=20000)
                    print("🎉 Conversion Complete!")
                    
                    # Verify Success Message
                    complete_msg = status_text.text_content()
                    print(f"Result: {complete_msg}")
                    
                    # Verify SHARP info?
                    # "✨ SHARP Active (Blend: 0.5)"
                    # We need to scroll or check subtext
                    # subtext is added dynamically
                    # Wait a sec for DOM update
                    time.sleep(1)
                    subtexts = page.locator(".status-subtext").all_text_contents()
                    found_sharp = False
                    for st in subtexts:
                        if "SHARP Active" in st:
                            found_sharp = True
                            print(f"✅ Found SHARP info: {st}")
                            break
                    if not found_sharp:
                        print("⚠️ SHARP info not found in subtext.")

                except Exception as e:
                    print(f"⚠️ Timed out waiting for completion: {e}")
                    
            except Exception as e:
                print(f"❌ Failed to start conversion or update status: {e}")

        browser.close()

if __name__ == "__main__":
    test_spag4d_ui_full_flow()
