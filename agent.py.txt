from agents import Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
from pydantic import BaseModel

cartas_atendente = Agent(
  name="cartas_atendente",
  instructions="Você é o atendente secundário do produto cartas bíblicas. O seu propósito é somente analisar as respostas dos clientes após cada parte do funil e entender se eles entenderam 100% a informação, caso tenham entendido você deve responder \"okay\" e caso você perceba que eles não entenderam, você deve responder \"duvida\"",
  model="gpt-4.1",
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


class WorkflowInput(BaseModel):
  input_as_text: str


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):
  with trace("cartas_atendente"):
    workflow = workflow_input.model_dump()
    conversation_history: list[TResponseInputItem] = [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": workflow["input_as_text"]
          }
        ]
      }
    ]
    cartas_atendente_result_temp = await Runner.run(
      cartas_atendente,
      input=[
        *conversation_history
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_698ec1fc5a1481909556959f8a03004106c08e5160c00c46"
      })
    )

    conversation_history.extend([item.to_input_item() for item in cartas_atendente_result_temp.new_items])

    cartas_atendente_result = {
      "output_text": cartas_atendente_result_temp.final_output_as(str)
    }
