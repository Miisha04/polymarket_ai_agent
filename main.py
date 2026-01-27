from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="AI Agent Service")

# Инициализируем локальную модель через Ollama
llm = OllamaLLM(model="llama3.2") 

# Создаем простую цепочку (Chain) вместо сложного агента с инструментами
prompt = ChatPromptTemplate.from_template("""
You are helpful assistent.
User question: {input}
Answer:
""")

chain = prompt | llm

class AgentQuery(BaseModel):
    query: str

@app.post("/process")
async def process(data: AgentQuery):
    # Генерация ответа локальной моделью
    result = chain.invoke({"input": data.query})
    return {"output": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
