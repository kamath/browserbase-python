# %%
from dotenv import load_dotenv
import os
from browserbase import Browserbase

load_dotenv()

BROWSERBASE_API_KEY = os.environ["BROWSERBASE_API_KEY"]
BROWSERBASE_PROJECT_ID = os.environ["BROWSERBASE_PROJECT_ID"]

bb = Browserbase(api_key=BROWSERBASE_API_KEY, project_id=BROWSERBASE_PROJECT_ID)


# %%
@bb.async_playwright
async def search_ebay(browser, query: str):
    page = await browser.new_page()
    await page.goto(f"https://www.ebay.com/sch/i.html?_from=R40&_nkw={query}&_sacat=0")
    await page.wait_for_load_state("networkidle")

    results = []
    items = await page.query_selector_all(".s-item")
    for item in items[:20]:
        title = await item.query_selector(".s-item__title")
        price = await item.query_selector(".s-item__price")
        link = await item.query_selector(".s-item__link")
        imageWrapper = await item.query_selector(".s-item__image-wrapper")
        image = await imageWrapper.query_selector("img")

        if title and price and link:
            results.append(
                {
                    "title": await title.inner_text(),
                    "price": await price.inner_text(),
                    "link": await link.get_attribute("href"),
                    "image": await image.get_attribute("src") if image else None,
                }
            )

    return results


ebay_results = await search_ebay("clothes")
ebay_results


# %%
@bb.selenium
def search_poshmark(driver, query: str):
    driver.get(f"https://poshmark.com/search?query={query}&type=listings&src=dir")
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.tiles_container"))
    )

    results = []
    items = driver.find_elements(
        By.CSS_SELECTOR, "div[data-et-prop-location='listing_tile']"
    )
    for item in items[:20]:
        try:
            try:
                title = item.find_element(By.CSS_SELECTOR, ".tile__title").text
            except:
                title = None
            try:
                price = item.find_element(By.CSS_SELECTOR, ".fw--bold").text
            except:
                price = None
            try:
                link = item.find_element(
                    By.CSS_SELECTOR, "a.tile__covershot"
                ).get_attribute("href")
            except:
                link = None
            try:
                imageWrapper = item.find_element(By.CSS_SELECTOR, ".img__container")
                image = imageWrapper.find_element(By.CSS_SELECTOR, "img").get_attribute(
                    "src"
                )
            except:
                image = None

            results.append(
                {"title": title, "price": price, "link": link, "image": image}
            )
        except Exception as e:
            print(f"Error processing item: {str(e)}")
            continue

    return results


poshmark_results = search_poshmark("handbags")
poshmark_results
# %%
