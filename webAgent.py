from typing import Dict, List
from autogen import ConversableAgent
import sys
import os
import re
import praw
import requests

# Reddit Fetching Function
def fetch_car_data(car_model: str) -> Dict[str, List[str]]:
    print(f"Fetching car data for: {car_model}")
    normalized_input = re.sub(r'[^a-zA-Z0-9]', '', car_model).lower()

    reddit = praw.Reddit(
        client_id='lw-HvQk3g6qXmAh02Qjo_A', 
        client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
        user_agent="car_review_scraper"
    )

    subreddit = reddit.subreddit("cars")
    reviews = []
    seen_titles = set()

    for submission in subreddit.search(car_model, limit=20):
        normalized_title = re.sub(r'[^a-zA-Z0-9]', '', submission.title).lower()
        if normalized_input in normalized_title and submission.title not in seen_titles:
            seen_titles.add(submission.title)
            if submission.selftext.strip():
                reviews.append(submission.selftext.strip())
            elif submission.title.strip():
                reviews.append(submission.title.strip())
            submission.comments.replace_more(limit=0)
            for comment in submission.comments[:3]:
                reviews.append(comment.body.strip())
    reviews = [r for r in reviews if r]
    print(f"Found {len(reviews)} Reddit reviews.")
    return {car_model: reviews}

# Bing Web Search API Integration
def fetch_web_data(car_model: str) -> Dict[str, List[str]]:
    print(f"Fetching web search results for: {car_model}")
    bing_api_key = os.environ.get("BING_API_KEY")
    endpoint = "https://api.bing.microsoft.com/v7.0/search"

    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    params = {"q": f"{car_model} reviews opinions pros cons", "count": 5}

    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()

    data = response.json()
    summaries = []

    # Extract content from search results
    for web_result in data.get("webPages", {}).get("value", []):
        summaries.append(f"{web_result['name']}: {web_result['snippet']}")
    
    print(f"Found {len(summaries)} web search results.")
    return {car_model: summaries}

def summarize_reviews_via_llm(reviews: List[str], llm_config) -> List[str]:
    from autogen import OpenAIWrapper

    # Initialize OpenAIWrapper
    model = OpenAIWrapper(api_key=os.environ.get("OPENAI_API_KEY"), model=llm_config["config_list"][0]["model"])

    # Construct prompt
    prompt = f"""
    Here are reviews and opinions about a car model:
    {reviews}

    Summarize the key points about performance, features, reliability, and user feedback.
    """

    # Call the create method with required arguments
    response = model.create(
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract response content
    content = response.choices[0].message.content
    summary = content.split("\n")

    return [line.strip() for line in summary if line.strip()]

# Agents Setup
def create_entrypoint_agent(llm_config):
    system_message = """
    You are the main coordinator. Fetch car reviews from Reddit and the web.
    Combine results and summarize insights for the user.
    """
    return ConversableAgent("entrypoint_agent", system_message=system_message, llm_config=llm_config)

def main(user_query: str):
    llm_config = {"config_list": [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}]}

    try:
        # Fetch data from Reddit
        reddit_result = fetch_car_data(user_query)

        # Fetch data from Web
        web_result = fetch_web_data(user_query)

        # Combine results
        all_reviews = reddit_result[user_query] + web_result[user_query]
        print(f"\nTotal reviews collected: {len(all_reviews)}")

        # Summarize reviews via GPT-4
        summarized_reviews = summarize_reviews_via_llm(all_reviews, llm_config)

        # Print Final Summary
        print("\nFinal Summary of Reviews:")
        for line in summarized_reviews:
            print(f"- {line}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please provide a query for a car model."
    main(sys.argv[1])
