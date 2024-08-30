# %%
# !pip install instructor pandas openai anthropic scikit-learn tqdm

# %%
from dotenv import load_dotenv
import os
from browserbase import Browserbase
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# %%
load_dotenv()

BROWSERBASE_API_KEY = os.environ["BROWSERBASE_API_KEY"]
BROWSERBASE_PROJECT_ID = os.environ["BROWSERBASE_PROJECT_ID"]
PPLX_API_KEY = os.environ["PPLX_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
bb = Browserbase(api_key=BROWSERBASE_API_KEY, project_id=BROWSERBASE_PROJECT_ID)

# %%
user_description = "It is my friend's birthday. He is 25 years old, and he likes to wear sneakers to the gym and to run errands. He is 5'10, 160 lbs, and he has a athletic build."

# %%

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


def ask_ai(model="claude-3-5-sonnet-20240620"):
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

    messages = [{"role": "user", "content": prompt}]

    try:
        # Anthropic w Instructor
        data = client.chat.completions.create(
            max_tokens=1024, model=model, messages=messages, response_model=SearchQuery
        )
        return data
    except Exception as e:
        # Anthropic w/o Instructor
        message = llm.messages.create(model=model, max_tokens=1024, messages=messages)
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


# ebay_results = await search_ebay("clothes")
# ebay_results_df = pd.DataFrame(ebay_results)
# ebay_results_df


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

    # Remove any results that have None values
    return [a for a in results if not all(x is None for x in a.values())]


# poshmark_results = search_poshmark("handbags")
# poshmark_results_df = pd.DataFrame(poshmark_results)
# poshmark_results_df
# %%


async def search(query: str):
    ebay_results = await search_ebay(query)
    ebay_results_df = pd.DataFrame(ebay_results)
    ebay_results_df["source"] = "ebay"
    poshmark_results = search_poshmark(query)
    poshmark_results_df = pd.DataFrame(poshmark_results)
    poshmark_results_df["source"] = "poshmark"
    return pd.concat([ebay_results_df, poshmark_results_df])


# results = await search("handbags")
# results

# %%


def vectorize_results(query, results):
    corpus = np.array([query] + list(results["title"]))
    # count_vectorizer = CountVectorizer(stop_words='english', min_df=0.005)
    # corpus2 = count_vectorizer.fit_transform(corpus)
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", use_idf=True)
    corpus2 = tfidf_vectorizer.fit_transform(corpus)
    df_tfidf = pd.DataFrame(
        corpus2.toarray(), columns=tfidf_vectorizer.get_feature_names_out()
    )

    search_df = pd.DataFrame(
        [
            *[
                df_tfidf[word]
                for word in query.strip().lower().split(" ")
                if word in df_tfidf.columns
            ],
            sum(
                [
                    df_tfidf[word]
                    for word in query.strip().lower().split(" ")
                    if word in df_tfidf.columns
                ]
            ),
        ],
        index=[
            *[
                word
                for word in query.strip().lower().split(" ")
                if word in df_tfidf.columns
            ],
            "Full Query",
        ],
    ).T
    search_df.index -= 1
    search_df = search_df.iloc[1:]
    return search_df["Full Query"]


# %%
async def full_pipeline(k_queries=5):
    queries = ask_ai()
    print("Attempting to Find:", queries.attempting_to_find)
    print("Additional Keywords:", queries.additional_keywords)
    print("Queries:", queries.search_queries)
    query_results = []
    for query in tqdm(queries.search_queries[:k_queries]):
        results = await search(query)
        results_scored = vectorize_results(query, results)
        results_scored = pd.merge(
            results, results_scored, left_index=True, right_index=True
        )
        results_scored["Search Query"] = query
        results_scored["Original Index"] = results_scored.index
        query_results.append(results_scored)
    full_query_results = pd.concat(query_results)
    size_series = full_query_results.groupby(["link"]).size().reset_index(name="count")
    results_ranked = (
        pd.merge(full_query_results, size_series, on="link", how="left")
        .reset_index()
        .sort_values(["Full Query", "Original Index"])
    )
    return results_ranked


await full_pipeline(1)

# %%
bb.sessions

# %%
bb.get_session_recording(bb.sessions["search_ebay"])
# %%
