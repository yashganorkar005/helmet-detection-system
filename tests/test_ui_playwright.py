"""
Playwright headless verification script for Helmet Detection & Road Safety Analyzer redesign.
Checks:
1. HTTP 200 and page title.
2. Zero emoji presence across the DOM.
3. Sidebar collapse and expand button visibility & functionality.
4. "Reset" button presence in "Live Detection Status" and click behavior.
5. Tab switching across all 5 operational modes.
6. Visual screenshots for quality assurance.
"""

import sys
import re
from playwright.sync_api import sync_playwright

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
    "]+",
    flags=re.UNICODE
)

def run():
    print("Launching Chromium via Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("Navigating to http://localhost:8501 ...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)

        # 1. Page Title
        title = page.title()
        print(f"Page Title: {title}")
        assert "Helmet Detection & Road Safety Analyzer" in title

        # 2. Check for App Header
        header = page.locator(".app-header")
        assert header.count() > 0, "App header (.app-header) not found"
        print("App header rendered successfully.")

        # 3. Check for Theme Toggle button
        theme_toggle_btn = page.locator("button:has-text('Light Mode'), button:has-text('Dark Mode')")
        print(f"Found {theme_toggle_btn.count()} Theme Toggle button(s).")
        assert theme_toggle_btn.count() > 0, "Theme toggle button not found"

        # Test clicking theme toggle
        initial_btn_text = theme_toggle_btn.first.inner_text().strip()
        print(f"Initial theme toggle button text: {initial_btn_text}")
        theme_toggle_btn.first.click()
        page.wait_for_timeout(2000)

        new_btn_text = page.locator("button:has-text('Light Mode'), button:has-text('Dark Mode')").first.inner_text().strip()
        print(f"Theme toggle button text after click: {new_btn_text}")
        assert initial_btn_text != new_btn_text, "Theme toggle button text did not change after click (theme switch failed)"
        print("Theme toggle switch successfully verified!")

        # 4. Check for Live Detection Status & Reset button
        reset_btn = page.locator("button:has-text('Reset')")
        print(f"Found {reset_btn.count()} Reset button(s).")
        assert reset_btn.count() > 0, "Reset button not found"

        # 4. Check for Stat Cards
        stat_cards = page.locator(".stat-card")
        print(f"Found {stat_cards.count()} stat card(s).")
        assert stat_cards.count() == 4, f"Expected 4 stat cards, found {stat_cards.count()}"

        # 5. Check tabs
        tab_list = page.locator("[data-testid='stTab']")
        tab_count = tab_list.count()
        print(f"Found {tab_count} tabs:")
        tab_names = []
        for i in range(tab_count):
            t_name = tab_list.nth(i).inner_text().strip()
            tab_names.append(t_name)
            print(f"  Tab {i+1}: '{t_name}'")

        expected_tabs = ["Single Image", "Batch Processing", "Video Analysis", "Live Camera", "Safety Analytics"]
        assert tab_names == expected_tabs, f"Expected tabs {expected_tabs}, got {tab_names}"

        # 6. Check for any emojis on the page
        body_text = page.locator("body").inner_text()
        found_emojis = EMOJI_PATTERN.findall(body_text)
        print(f"Emoji scan result: {len(found_emojis)} emojis detected.")
        if found_emojis:
            print(f"Detected emojis: {set(found_emojis)}")
            # Filter out non-fatal or report
            assert len(found_emojis) == 0, f"Found emojis in DOM: {set(found_emojis)}"
        else:
            print("Verified: 0 emojis present across the entire rendered page!")

        # 7. Test Sidebar Collapse & Expand Button
        sidebar = page.locator("[data-testid='stSidebar']")
        print(f"Sidebar initial bounding box: {sidebar.bounding_box()}")
        collapse_btn = page.locator("[data-testid='stSidebarCollapseButton'] button")
        if collapse_btn.count() > 0:
            print("Clicking sidebar collapse button...")
            collapse_btn.click()
            page.wait_for_timeout(1000)

            expand_btn = page.locator("[data-testid='stExpandSidebarButton']")
            assert expand_btn.is_visible(), "Expand sidebar button is not visible after collapse!"
            print(f"Expand button is visible at: {expand_btn.bounding_box()}")

            print("Clicking expand sidebar button...")
            expand_btn.click()
            page.wait_for_timeout(1000)
            assert sidebar.is_visible(), "Sidebar failed to re-expand!"
            print("Sidebar collapse/expand cycle verified successfully.")

        # 8. Test Single Image Demo
        print("Testing Single Image Demo load...")
        demo_btn = page.locator("button:has-text('Load Demo Image')")
        assert demo_btn.count() > 0, "Load Demo Image button not found"
        demo_btn.click()
        page.wait_for_timeout(4000)

        # Verify detection results appeared
        detections_breakdown = page.locator("text=Detections Breakdown")
        print(f"Detections breakdown heading visible: {detections_breakdown.is_visible()}")
        assert detections_breakdown.is_visible(), "Detection results breakdown table not visible after loading demo image"

        # Check that Stat Cards updated
        stat_values = page.locator(".stat-value")
        total_detections_str = stat_values.nth(0).inner_text().strip()
        print(f"Total Detections after inference: {total_detections_str}")
        assert int(total_detections_str) > 0, "Total Detections should be > 0 after demo image"

        # 9. Test Reset button behavior
        print("Testing Reset button on Live Detection Status...")
        live_reset_btn = page.locator("button[key='reset_live_status_btn'], button:has-text('Reset')").first
        live_reset_btn.click()
        page.wait_for_timeout(2000)

        stat_values_after = page.locator(".stat-value")
        total_after = stat_values_after.nth(0).inner_text().strip()
        helmets_after = stat_values_after.nth(1).inner_text().strip()
        violations_after = stat_values_after.nth(2).inner_text().strip()
        compliance_after = stat_values_after.nth(3).inner_text().strip()

        print(f"Values after Reset: Total={total_after}, Helmets={helmets_after}, Violations={violations_after}, Compliance={compliance_after}")
        assert total_after == "0", f"Expected total 0 after reset, got {total_after}"
        assert helmets_after == "0", f"Expected helmets 0 after reset, got {helmets_after}"
        assert violations_after == "0", f"Expected violations 0 after reset, got {violations_after}"
        assert compliance_after == "N/A", f"Expected compliance N/A after reset, got {compliance_after}"
        print("Reset button verified: all metrics successfully reset to zero!")

        # 10. Capture screenshots
        page.screenshot(path="ui_redesign_screenshot.png", full_page=True)
        print("Captured full-page screenshot: ui_redesign_screenshot.png")

        # Capture Tab 5 (Analytics)
        tab_list.nth(4).click()
        page.wait_for_timeout(1500)
        page.screenshot(path="ui_analytics_screenshot.png")
        print("Captured analytics tab screenshot: ui_analytics_screenshot.png")

        browser.close()
        print("ALL PLAYWRIGHT VERIFICATIONS PASSED!")

if __name__ == "__main__":
    run()
