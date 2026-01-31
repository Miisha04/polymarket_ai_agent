from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from fastapi.concurrency import run_in_threadpool

app = FastAPI(title="AI Agent Service")

# Локальная модель Ollama
llm = OllamaLLM(model="llama3.2") 

# Простая цепочка prompt → llm
base_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Here is the conversation so far:
{history}

User question: {input}
Answer:
""")
chain = base_prompt | llm

# Входной формат
class AgentQuery(BaseModel):
    input: str
    history: list[dict] | None = None  # [{"role": "user"/"assistant", "content": "..."}]

@app.post("/process")
async def process(data: AgentQuery):
    # Формируем текстовую историю для prompt
    history_text = ""
    if data.history:
        for msg in data.history:
            history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    prompt_input = {
        "input": data.input,
        "history": history_text.strip()
    }

    try:
        # Асинхронный вызов LLM через threadpool
        result = await run_in_threadpool(chain.invoke, prompt_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    return {"output": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
