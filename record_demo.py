#!/usr/bin/env python3
"""
TerrainGuard – Demo Video Recorder
===================================
Uses Playwright to automate the web dashboard and record a demo video
showcasing all key features: map navigation, MAYDAY triage, flight routing.
"""

import os
import time
from playwright.sync_api import sync_playwright

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(SCRIPT_DIR, "index.html")
VIDEO_DIR = os.path.join(SCRIPT_DIR, "demo_video")


def record_demo():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=VIDEO_DIR,
            record_video_size={"width": 1440, "height": 900},
        )
        page = context.new_page()

        # ── Load the dashboard ──
        page.goto(f"file://{HTML_PATH}")
        page.wait_for_load_state("networkidle")
        time.sleep(3)  # Let map tiles load

        # ── Scene 1: Pan around the map ──
        print("[1/5] Showing map overview …")
        # Zoom in slightly on the terrain
        page.mouse.move(720, 450)
        for _ in range(2):
            page.mouse.wheel(0, -150)
            time.sleep(0.5)
        time.sleep(2)

        # ── Scene 2: Toggle LULC layer on ──
        print("[2/5] Toggling LULC overlay …")
        page.click("#toggle-lulc", force=True)
        time.sleep(2.5)

        # Toggle LULC back off
        page.click("#toggle-lulc", force=True)
        time.sleep(1)

        # ── Scene 3: Activate MAYDAY ──
        print("[3/5] Activating MAYDAY triage …")
        page.click("#btn-mayday")
        time.sleep(4)  # Let animations play out

        # Scroll sidebar to show triage cards
        page.evaluate("document.getElementById('sidebar').scrollTop = 300")
        time.sleep(2)

        # Scroll further to show sector table
        page.evaluate("document.getElementById('sidebar').scrollTop = 700")
        time.sleep(2.5)

        # Scroll back to top of sidebar
        page.evaluate("document.getElementById('sidebar').scrollTop = 0")
        time.sleep(1)

        # ── Scene 4: Click on a triage card to fly to landing site ──
        print("[4/5] Inspecting landing vectors …")
        page.evaluate("document.getElementById('sidebar').scrollTop = 350")
        time.sleep(1)
        # Click second triage card
        triage_cards = page.query_selector_all(".triage-card")
        if len(triage_cards) >= 2:
            triage_cards[1].click()
            time.sleep(3)

        # Zoom back out
        page.mouse.move(720, 450)
        for _ in range(3):
            page.mouse.wheel(0, 150)
            time.sleep(0.3)
        time.sleep(2)

        # ── Scene 5: Deactivate MAYDAY and add custom waypoints ──
        print("[5/5] Custom flight routing …")
        page.click("#btn-mayday")  # Deactivate
        time.sleep(1)

        # Clear existing waypoints
        page.click("#btn-clear-wp")
        time.sleep(0.5)

        # Enter drawing mode
        page.click("#btn-add-wp")
        time.sleep(0.5)

        # Click several points on the map to create a custom route
        map_el = page.query_selector("#map")
        box = map_el.bounding_box()
        if box:
            points = [
                (box["x"] + box["width"] * 0.15, box["y"] + box["height"] * 0.7),
                (box["x"] + box["width"] * 0.35, box["y"] + box["height"] * 0.5),
                (box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.45),
                (box["x"] + box["width"] * 0.75, box["y"] + box["height"] * 0.3),
            ]
            for px, py in points:
                page.mouse.click(px, py)
                time.sleep(0.8)

        # Exit drawing mode
        page.click("#btn-add-wp")
        time.sleep(2)

        # Final pause
        time.sleep(2)

        # ── Close ──
        context.close()
        browser.close()

    # Find the recorded video file
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".webm")]
    if video_files:
        src = os.path.join(VIDEO_DIR, video_files[0])
        dst = os.path.join(SCRIPT_DIR, "demo.webm")
        os.rename(src, dst)
        print(f"\n[✓] Demo video saved: {dst}")
        # Clean up temp dir
        try:
            os.rmdir(VIDEO_DIR)
        except OSError:
            pass
        return dst
    else:
        print("[!] No video file found.")
        return None


if __name__ == "__main__":
    print("TerrainGuard – Recording Demo Video …\n")
    path = record_demo()
    if path:
        print(f"Video ready at: {path}")
