#%%
# !pip install instructor pandas openai anthropic

# %%
from dotenv import load_dotenv
import os
from browserbase import Browserbase
import json
import requests

load_dotenv()

BROWSERBASE_API_KEY = os.environ["BROWSERBASE_API_KEY"]
BROWSERBASE_PROJECT_ID = os.environ["BROWSERBASE_PROJECT_ID"]
PPLX_API_KEY = os.environ["PPLX_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
bb = Browserbase(api_key=BROWSERBASE_API_KEY, project_id=BROWSERBASE_PROJECT_ID)

#%%
user_description = "I am looking for a pair of sneakers for my friend for his birthday. He is 25 years old, and he likes to wear sneakers to the gym and to run errands. He is 5'10, 160 lbs, and he has a athletic build."

#%%

from openai import OpenAI
import instructor
from pydantic import BaseModel
from typing import List
from anthropic import Anthropic

class SearchQuery(BaseModel):
	attempting_to_find: str
	additional_keywords: List[str]
	search_queries: List[str]

# llm = OpenAI(api_key=PPLX_API_KEY, base_url="https://api.perplexity.ai")
llm = Anthropic(api_key=ANTHROPIC_API_KEY)
# client = instructor.from_openai(llm)
client = instructor.from_anthropic(Anthropic())

def ask_ai(model="llama-3.1-sonar-large-128k-chat"):
	prompt = f"""
		You are an expert consignment store owner helping a user find clothes based on a certain vibe or profile.
		Generate a list of keywords and phrases that match specific clothing items that match the vibe of "{user_description}"
		The queries should be for specific items of clothing like "sneaker", "dress", "sweater", etc. with specific objective descriptors, avoiding anything subjective.
		The queries should be able to be directly inputted into a search engine on ebay or poshmark to get results.
		Return data in JSON format in the following structure:
		{{"attempting_to_find": x, "additional_keywords": y, "search_queries": z}}
		Where attempting_to_find is a general few words describing to the user what they are searching for in plain english,
		additional_keywords is a list of phrases that can be appended to queries in search_queries to get more specific by price point, formality, vibe, etc.,
		and search_queries is the list of queries generated.
		Order search_queries with the most relevant queries first
	"""
	
	messages = [
			{
				"role": "user",
				"content": prompt
			}
		]

	try:
		# Anthropic w Instructor
		data = client.chat.completions.create(max_tokens=1024, model=model, messages=messages, response_model=SearchQuery)
		return data
	except Exception as e:
		# Anthropic w/o Instructor
		message = llm.messages.create(
			model=model,
			max_tokens=1024,
			messages=messages
		)
		data_wo_instructor = json.loads(message.content[0].text)
		return SearchQuery(**data_wo_instructor)

res = ask_ai()
res

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

	return [a for a in results if not all(x is None for x in a.values())]


poshmark_results = search_poshmark("handbags")
poshmark_results
# %%
!pip install pandas
import pandas as pd

poshmark_results_df = pd.DataFrame(poshmark_results)
poshmark_results_df
# %%

