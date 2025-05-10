from doctest import FAIL_FAST
import enum
import json
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from loguru import logger
import openai
from openai.types.chat import ChatCompletion

from brain.settings import get_settings


def connect_to_openai():
    """
    Connects to the OpenAI API and returns a simple response.
    Uses settings from the environment variables via Pydantic.
    """
    settings = get_settings()

    # Initialize the client with API key from settings
    client = openai.OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url="https://openrouter.ai/api/v1",
    )

    # Make a simple request to the API
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, OpenAI!"},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}"


def add_tool_message(
    tool_calls: List[Any],
    tool_results: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
) -> None:
    """
    Add tool call and result messages to the conversation history.

    Args:
        tool_calls: List of tool calls from the assistant
        tool_results: List of results from executing the tools
        messages: Conversation history to update
    """
    # Add the assistant's tool call
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
    )

    # Add the tool results
    for tool_call, result in zip(tool_calls, tool_results):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": str(result),
            }
        )


def call_with_tools(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    tool_choice: Optional[Union[str, Dict[str, str]]] = "auto",
) -> ChatCompletion:
    """
    Call the OpenAI API with tool definitions.

    Args:
        messages: List of message dictionaries with role and content
        tools: List of tool definitions
        tool_choice: Controls tool calling behavior ("auto", "none", or specific tool)

    Returns:
        The full ChatCompletion response
    """
    settings = get_settings()
    client = openai.OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url="https://openrouter.ai/api/v1",
    )

    return client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def execute_tool_call(
    response: ChatCompletion,
    available_tools: Dict[str, Callable],
    messages: List[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """
    Execute a tool call from an OpenAI response.
    """
    message = response.choices[0].message
    #    logger.info(f"Tool Arguments: {message.tool_calls}")
    # Check if the model wants to call a tool
    if hasattr(message, "tool_calls") and message.tool_calls:
        tool_results = []

        try:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                logger.info(f"\n=== Executing Tool: {tool_name} ===")
                logger.info(f"Tool ID: {tool_call.id}")  # Log the tool ID

                if tool_name not in available_tools:
                    logger.error(f"Tool not found: {tool_name}")
                    return False

                tool_args = json.loads(tool_call.function.arguments)
                logger.info(f"Tool Arguments: {tool_args}")

                tool_to_call = available_tools[tool_name]
                result = tool_to_call["function"](**tool_to_call["parameters"])
                logger.info(f"Tool Result: {result}")

                tool_results.append(result)

                if not result.get("status", False):
                    error_msg = (
                        f"Error: Tool {tool_name} failed, all further tools are skipped"
                    )
                    logger.error(error_msg)
                    add_tool_message(message.tool_calls, tool_results, messages)
                    return False

            # Add successful tool results to conversation history
            add_tool_message(message.tool_calls, tool_results, messages)
            logger.info("All tools executed successfully")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Execution error: {str(e)}")
            return False

    return False
