from doctest import FAIL_FAST
from pydantic import BaseModel
from brain.fast_api import app
from brain.controller import controller
from collections import deque
from brain.connection import call_with_tools, execute_tool_call
from brain.tools.tools import tools
from loguru import logger
import asyncio
#import pyttsx3
#engine = pyttsx3.init()
#voices = engine.getProperty('voices')
#for voice in voices:
#    if "german" in voice.name.lower() or "deutsch" in voice.name.lower():
#        engine.setProperty('voice', voice.id)
#        break

#engine.setProperty('rate', 175)
#engine.setProperty('volume', 0.9)

class MessageRequest(BaseModel):
    message: str


messages = deque(maxlen=100)
message_lock = asyncio.Lock()


@app.post("/stop")
async def stop():
    controller.stop_event()
    return "ok"


@app.post("/resume")
async def resume():
    controller.resume_from_stop()
    return "ok"


@app.post("/command")
async def receive_message(request: MessageRequest):
    async with message_lock:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, help_receive_message, request)
        return result


def help_receive_message(request: MessageRequest):
    try:

        logger.info(f"User Input: {request.message}")
        messages.append({"role": "user", "content": request.message})
        iteration = 0
        while messages[0]["role"] == "tool" or messages[0]["role"] == "assistant":
            messages.popleft()
        while True:
            iteration += 1
            logger.info(f"\n=== Iteration {iteration} ===")
            response = call_with_tools(messages, tools)
            model_message = response.choices[0].message

            logger.info(
                f"Model Response: {model_message.content if model_message.content else 'No content (tool call)'}"
            )
            # Execute tool if called
            tool_result = execute_tool_call(
                response, controller.create_tool_mapping(), messages
            )
            if not tool_result:
                break
        
#        engine.say(response.choices[0].message.content)
#        engine.runAndWait()
        # Return the final response after all tool executions
        return {"message": response.choices[0].message.content}

    except Exception as e:
        return {"error": f"Error processing message: {str(e)}"}


@app.post("/interrupt")
async def interrupt():
    controller.interrupt()
    return {"status": "resumed"}
