from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
# from src.utils.helpers import read_json

import json
def read_json(file_path):
    try:
        f = open(file_path)
        data = json.load(f)
        f.close()
    except Exception as e:
        print(e)
        data = []
    return data

model_name = "Qwen/Qwen3-1.7B"
device = "cuda" if torch.cuda.is_available() else "cpu"

clean_preview_data = read_json("/mnt/c/Users/Administrator/PycharmProjects/arkou/misc_files/clean_preview_data.json")


def get_top_restaurants(data, top_n=5):
    # Filter only restaurants
    restaurants = []
    for place in data:
        if isinstance(place, dict):
            if 'type' in list(place.keys()):
                if isinstance(place['type'], str):
                    if re.search("restaurant", place['type']):
                        if 'reviews_score' not in list(place.keys()):
                            place['reviews_score'] = 0
                        if 'reviews_count' not in list(place.keys()):
                            place['reviews_count'] = 0
                        # if 'description' not in list(place.keys()):
                        #     place['description'] = "Nothing yet"
                        print(place['name'],place['type'],place['reviews_score'],place['reviews_count'])
                        restaurants.append(place)

    print("restaurants: ", len(restaurants))
    
    # Sort by rating (highest first) and then by number of reviews
    sorted_restaurants = sorted(restaurants, key=lambda x: (x['reviews_score'], x['reviews_count']), reverse=True)
    
    # Select top N
    top_restaurants = sorted_restaurants[:top_n]
    
    # Format the output
    response = "Here are the top-rated restaurants in Djerba:\n"
    for i, place in enumerate(top_restaurants, start=1):
        # response += f"{i}. {place['name']} (Rating: {place['rating']}, Reviews: {place['reviews']}) - {place['description']}\n"
        response += f"{i}. {place['name']} (Rating: {place['reviews_score']}, Reviews: {place['reviews_count']})\n"

    return response

# Example usage:


# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

def query_qwen(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**model_inputs, max_new_tokens=256, temperature=0.7)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Construct the prompt with context
user_input = "Show me the top 5 restaurants in Djerba."
context = get_top_restaurants(clean_preview_data)

# Create the final prompt
prompt = f"User asked: {user_input}\n\nBased on the data, here are some top-rated restaurants:\n{context}\n\nNow, please provide a more personalized suggestion for the user."

# Query Qwen
response = query_qwen(prompt)
print(response)
