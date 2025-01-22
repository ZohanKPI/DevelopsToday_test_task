import faiss
import os
import pandas as pd
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=api_key, 
)
app = FastAPI(title="Cocktail Advisor Chat", description="A cocktail chat app.", version="1.0.0")
templates = Jinja2Templates(directory="templates")


cocktail_data = pd.read_csv("datasets/final_cocktails.csv")

template = """
Provide advice regarding cocktails:
{preference_context}
{cocktail_context}
Answer the following question or give a recommendation: {query}
"""
prompt_template = PromptTemplate(
    input_variables=["preference_context", "cocktail_context", "query"],
    template=template
)
def query_gpt(prompt):
    try:
        chat_completion = client.chat.completions.create(
        messages=[
        {
            "role": "user",
            "content": prompt,
        }
        ],
        model="gpt-3.5-turbo",
)
        response_text = chat_completion.choices[0].message.content.strip()
        return response_text
    except Exception as e:
        return f"Error: {str(e)}"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
cocktail_embeddings = embedding_model.encode(cocktail_data['text'].tolist())
dimension = cocktail_embeddings.shape[1]
cocktail_index = faiss.IndexFlatL2(dimension)
cocktail_index.add(cocktail_embeddings)

preferences = []
preference_list = ["i like", "i love", "i enjoy", "i prefer", "my favorite", "my beloved",
    "vodka", "rum", "gin", "tequila", "whiskey", "lime", "lemon",
    "mint", "sugar", "orange juice", "pineapple juice", "cranberry juice",
    "coconut cream", "triple sec", "vermouth", "bitters"
]
def search_cocktail_data(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = cocktail_index.search(query_embedding, top_k)

    relevant_cocktails = [
        cocktail_data['text'].iloc[i] for i in indices[0] if i < len(cocktail_data)
    ]
    return relevant_cocktails

def add_preference(preference_text):
    global preferences
    embedding = embedding_model.encode([preference_text])

    cocktail_index.add(embedding)
    preferences.append(preference_text)

def search_preferences(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = cocktail_index.search(query_embedding, top_k)

    results = [preferences[i] for i in indices[0] if i < len(preferences)]

    return results

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat/", response_class=HTMLResponse)
async def chat(request: Request, user_input: str):
    if not user_input.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if any(keyword in user_input.lower() for keyword in preference_list):
        add_preference(user_input)
    
    retrieved_preferences = search_preferences(user_input)
    relevant_cocktails = search_cocktail_data(user_input)
    preference_context = (
        f"The user has the following preferences: {', '.join(retrieved_preferences)}."
        if retrieved_preferences else ""
    )
    cocktail_context = (
        f"Here is some relevant cocktail information: {', '.join(relevant_cocktails)}."
        if relevant_cocktails else ""
    )
    enhanced_prompt = prompt_template.format(
        preference_context=preference_context,
        cocktail_context=cocktail_context,
        query=user_input
    )
    gpt_response = query_gpt(enhanced_prompt)
    return templates.TemplateResponse("index.html", {"request": request, "response": gpt_response})
