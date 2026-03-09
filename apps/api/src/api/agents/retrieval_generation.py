
from qdrant_client import QdrantClient

from openai import OpenAI
from api.core.config import config

openai_client = OpenAI(api_key=config.OPENAI_API_KEY)


def get_embedding(text, model="text-embedding-3-small"):
    # Use the openai_client instance created in Cell 1
    response = openai_client.embeddings.create(
        input=text,
        model=model,
    )
    return response.data[0].embedding


def retrieve_data(query,qdrant_client, k=5):
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-00",
        query=query_embedding,
        limit=k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        retrieved_context_ratings.append(result.payload["rating"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }


def process_context(context:dict):

    formatted_context = ""

    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context


def build_prompt(preprocessed_context, question):

    prompt = f"""
You are a movie recommendation assistant.

You will be given a question and a list of context.

Instructtions:
- You need to answer the question based on the provided context only.
- Never use word context and refer to it as the available products.

Context:
{preprocessed_context}

Question:
{question}
"""

    return prompt


def generate_answer(prompt):

    response = openai_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "system", "content": prompt}],
        reasoning_effort="minimal"
    )

    return response.choices[0].message.content


def rag_pipeline(question, top_k=5):

    qdrant_client = QdrantClient(url="http://localhost:6333")

    retrieved_context = retrieve_data(question, qdrant_client, top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)

    return answer

