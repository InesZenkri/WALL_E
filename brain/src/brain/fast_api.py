from fastapi import FastAPI
from loguru import logger
import sys

# Optional: Remove default logger handlers and redirect to loguru
import logging


# Intercept standard logging
class InterceptHandler(logging.Handler):
    def emit(self, record):
        logger_opt = logger.opt(depth=6, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())


# Configure root logging to use loguru
logging.basicConfig(handlers=[InterceptHandler()], level=0)

# Configure loguru
logger.remove()  # Remove default stderr
logger.add(sys.stderr, level="INFO")  # You can write to file too

# Import your tool call logic
from brain.connection import call_with_tools, execute_tool_call

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 FastAPI app is starting up...")


@app.get("/")
async def root():
    logger.info("Handling root endpoint.")
    return {"message": "Hello from FastAPI with Loguru!"}
