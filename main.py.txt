from fastapi import FastAPI
from pydantic import BaseModel
from agent import run_workflow, WorkflowInput

app = FastAPI()

class Payload(BaseModel):
    message: str

@app.post("/cartas")
async def cartas(payload: Payload):
    result = await run_workflow(WorkflowInput(input_as_text=payload.message))
    # força padronizar retorno como string simples
    reply = (result or "").strip().lower()
    if "duvida" in reply:
        reply = "duvida"
    elif "okay" in reply or "ok" in reply:
        reply = "okay"
    return {"reply": reply}
