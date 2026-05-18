import os
import asyncio
import uuid
import httpx
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
from PIL import Image
import io
import numpy as np

# List of 15 diseases prioritized for Black Skin
DISEASES = [
     "Post-Inflammatory Hyperpigmentation", "Acne Vulgaris", "Atopic Dermatitis",
    "Seborrheic Dermatitis", "Tinea Corporis", "Keloids", "Dermatosis Papulosa Nigra",
    "Pseudofolliculitis Barbae", "Vitiligo", "Pityriasis Alba", "Melasma",
    "Lichen Planus", "Traction Alopecia", "Acne Keloidalis Nuchae", "Acanthosis Nigricans"
]

# async def download_image(client, url, folder):
#     """Downloads an image from a URL and saves it with a unique ID."""
#     try:
#         response = await client.get(url, timeout=10)
#         if response.status_code == 200:
#             ext = url.split(".")[-1].split("?")[0] # Try to get extension
#             ext = ext if len(ext) < 5 else "jpg"
#             file_path = os.path.join(folder, f"{uuid.uuid4()}.{ext}")
#             with open(file_path, "wb") as f:
#                 f.write(response.content)
#     except Exception:
#         pass

async def download_image(client, url, folder):
    try:
        response = await client.get(url, timeout=10)
        if response.status_code == 200:
            # Open the image in memory to check skin tone
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            img.thumbnail((100, 100)) # Resize to 100x100 for fast processing
            
            # Convert to numpy array to get average color
            pixels = np.array(img)
            avg_color = np.mean(pixels, axis=(0, 1))
            
            # Simple "Melanin Threshold"
            # In RGB, higher values = lighter colors. 
            # We want to skip images that are too "bright/white" (e.g., R, G, B all > 200)
            if avg_color[0] > 210 and avg_color[1] > 190 and avg_color[2] > 180:
                # This is likely white skin or a white background diagram
                return False
            
            file_path = os.path.join(folder, f"{uuid.uuid4()}.jpg")
            with open(file_path, "wb") as f:
                f.write(response.content)
            return True
    except:
        return False

import random

async def scrape_disease(disease_name, num_images=150):
    folder = f"dataset/{disease_name.replace(' ', '_')}"
    os.makedirs(folder, exist_ok=True)
    
    async with async_playwright() as p:
        # Launch with headless=False so you can solve CAPTCHAs manually
        browser = await p.chromium.launch(headless=False)
        
        # Adding a more detailed context to look like a real browser
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        
        page = await context.new_page()
        await stealth_async(page)
        
        # Navigate to search
        # search_url = f"https://www.google.com/search?q={disease_name}+on+black+skin+patient&tbm=isch"
        # search_url = f"https://www.bing.com/images/search?q={disease_name}+on+black+skin+patient"
        # Change this:
        search_url = f"https://www.bing.com/images/search?q={disease_name}+on+black+skin+patient&form=HDRSC2"

        print(f"--- Navigating to: {disease_name} ---")
        await page.goto(search_url)

        # --- CAPTCHA CHECK ---
        # If you see the CAPTCHA, this will wait up to 60 seconds for YOU to click it.
        try:
            if await page.wait_for_selector("iframe[src*='recaptcha']", timeout=5000):
                print("!! CAPTCHA DETECTED !! Please solve it in the browser window now...")
                # The script stays here until the CAPTCHA is gone or 60s passes
                await page.wait_for_selector("iframe[src*='recaptcha']", state="hidden", timeout=60000)
                print("Continuing after CAPTCHA...")
        except:
            pass # No CAPTCHA found, moving on

        found_urls = set()
        last_height = await page.evaluate("document.body.scrollHeight")
        
        while len(found_urls) < num_images:
            # --- VARIABLE SCROLLING START ---
            # Instead of one big jump, we do 3-5 small "human" scrolls
            for _ in range(random.randint(3, 6)):
                scroll_step = random.randint(300, 800)
                await page.mouse.wheel(0, scroll_step)
                # Random tiny pauses between scroll steps
                await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Larger pause to let images render (mimics reading)
            await asyncio.sleep(random.uniform(2, 4))
            # --- VARIABLE SCROLLING END ---
            
            # # Robust extraction logic
            # new_urls = await page.evaluate("""() => {
            #     const imgs = Array.from(document.querySelectorAll('img'));
            #     return imgs
            #         .map(img => img.src || img.dataset.src || img.dataset.iurl)
            #         .filter(src => src && src.startsWith('http') && !src.includes('gstatic.com/favicon'));
            # }""")

            # # Replace your existing new_urls extraction with this:
            # new_urls = await page.evaluate("""() => {
            #     const results = [];
            #     // Bing uses the 'iusc' class for its image containers
            #     const elements = document.querySelectorAll('a.iusc');
            #     for (let el of elements) {
            #         try {
            #             // Bing stores high-res metadata in the 'm' attribute
            #             const m = JSON.parse(el.getAttribute('m'));
            #             if (m.murl) results.push(m.murl); 
            #         } catch (e) {}
            #     }
            #     return results;
            # }""")            
            
            # Updated extraction logic with keyword filtering
            new_urls = await page.evaluate("""() => {
                const results = [];
                const trashKeywords = ['vector', 'logo', 'cartoon', 'animated', 'infographic', 'text', 'diagram', 'chart'];
                const elements = document.querySelectorAll('a.iusc');
                
                for (let el of elements) {
                    try {
                        const m = JSON.parse(el.getAttribute('m'));
                        const url = m.murl.toLowerCase();
                        
                        // Skip if the URL contains any trash keywords
                        const isTrash = trashKeywords.some(keyword => url.includes(keyword));
                        
                        if (m.murl && !isTrash) {
                            results.push(m.murl); 
                        }
                    } catch (e) {}
                }
                return results;
            }""")


            found_urls.update(new_urls)
            print(f"[{disease_name}] Found {len(found_urls)} potential images...")

            if len(found_urls) >= num_images:
                break

            # Infinite scroll check
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                try:
                    # Click "Show more" if it exists
                    more_btn = await page.query_selector('input.mye4qd')
                    if more_btn:
                        await more_btn.click()
                        await asyncio.sleep(2)
                    else:
                        break
                except:
                    break
            last_height = new_height

        # Start downloading
        print(f"Downloading {min(len(found_urls), num_images)} images for {disease_name}...")
        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = [download_image(client, url, folder) for url in list(found_urls)[:num_images]]
            await asyncio.gather(*tasks)
        
        await browser.close()

async def main():
    for disease in DISEASES:
        print(f"\n--- SCRAPING: {disease} ---")
        await scrape_disease(disease)

if __name__ == "__main__":
    asyncio.run(main())
