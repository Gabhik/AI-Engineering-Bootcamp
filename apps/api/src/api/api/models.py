from pydantic import BaseModel, Field

class RAGRequest(BaseModel):
    query: str = Field(...,description="The query to be used in RAG pipeline")

class RAGResponse(BaseModel):
    answer: str = Field(...,description="The answer to the query")
    request_id: str = Field(...,description="The request ID")