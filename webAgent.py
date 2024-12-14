from typing import Dict, List
from autogen import ConversableAgent
import sys
import os
import re
import praw

def fetch_car_data(car_model: str) -> Dict[str, List[str]]:
    print(f"Fetching car data for: {car_model}")
    normalized_input = re.sub(r'[^a-zA-Z0-9]', '', car_model).lower()

    # Configure Reddit API client
    reddit = praw.Reddit(
        client_id='lw-HvQk3g6qXmAh02Qjo_A',  # Replace with your Reddit client_id
        client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
        user_agent="car_review_scraper"
    )

    subreddit = reddit.subreddit("cars")
    reviews = []
    seen_titles = set()  # To avoid duplicates

    # Search for posts related to the car model
    query = f"{car_model}"
    for submission in subreddit.search(query, limit=20):
        normalized_title = re.sub(r'[^a-zA-Z0-9]', '', submission.title).lower()
        if normalized_input in normalized_title and submission.title not in seen_titles:
            seen_titles.add(submission.title)

            # Check for meaningful content
            if submission.selftext.strip():
                reviews.append(submission.selftext.strip())
            elif submission.title.strip():
                reviews.append(submission.title.strip())
            
            # Optionally, include top comments
            submission.comments.replace_more(limit=0)
            for comment in submission.comments[:3]:  # Fetch up to 3 top comments
                reviews.append(comment.body.strip())

    # Filter out empty reviews
    reviews = [review for review in reviews if review]

    print(f"Found {len(reviews)} reviews for {car_model} from Reddit")
    return {car_model: reviews}

def get_data_fetch_agent_prompt(car_query: str) -> str:
    print(f"Generating prompt for data fetch agent with query: {car_query}")
    return f"""
    You are the data fetch agent. Your task is to analyze the query and determine the correct car model to fetch data for.
    Once you have identified the car model, use the fetch_car_data function to retrieve the reviews from Reddit.
    Query: {car_query}
    Please respond with a function call to fetch_car_data with the correct car model as the argument.
    """

def create_entrypoint_agent(llm_config):
    system_message = """
    You are the main entry point agent. You are responsible for coordinating the overall process of analyzing car reviews.
    Your task is to:
    1. Initiate the data fetch process with the data fetch agent.
    2. Provide a final response with the fetched reviews for the car.

    Do not engage in unnecessary conversation with other agents. Focus on moving through the steps efficiently.
    """
    agent = ConversableAgent("entrypoint_agent", system_message=system_message, llm_config=llm_config)
    return agent

def create_data_fetch_agent(user_query, llm_config):
    agent = ConversableAgent("data_fetch_agent", 
                             system_message=get_data_fetch_agent_prompt(user_query), 
                             llm_config=llm_config)
    agent.register_for_llm(name="fetch_car_data", description="Fetches the reviews for a specific car model from Reddit.")(fetch_car_data)
    agent.register_for_execution(name="fetch_car_data")(fetch_car_data)
    return agent

def main(user_query: str):
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}

    # Create and configure agents
    entrypoint_agent = create_entrypoint_agent(llm_config)
    data_fetch_agent = create_data_fetch_agent(user_query, llm_config)

    try:
        # Fetch car data
        print("Fetching car data...")
        data_fetch_result = data_fetch_agent.generate_reply(messages=[{"content": user_query, "role": "user"}])
        car_data = fetch_car_data(data_fetch_result['tool_calls'][0]['function']['arguments'].split('"')[3])
        car_model = list(car_data.keys())[0]
        reviews = car_data[car_model]
        print(f"Retrieved {len(reviews)} reviews for {car_model}")

        # Print final response
        print(f"Reviews for {car_model}:")
        for review in reviews:
            print(f"- {review}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Debug information:")
        print(f"User query: {user_query}")
        print(f"Data fetch result: {data_fetch_result if 'data_fetch_result' in locals() else 'Not available'}")

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for a car model when executing main."
    main(sys.argv[1])
