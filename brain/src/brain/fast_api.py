from contextlib import asynccontextmanager

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



@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("🚀 FastAPI app is starting up...")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    logger.info("Handling root endpoint.")
    return {"message": "Hello from FastAPI with Loguru!"}
