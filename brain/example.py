from typing import Dict, List

from brain.connection import call_with_tools, execute_tool_call
from brain.settings import get_settings


def search_database(query: str, limit: int = 5) -> List[Dict]:
    """
    Simulated database search function.

    Args:
        query: The search query
        limit: Maximum number of results to return

    Returns:
        List of search results
    """
    # This would normally query a real database
    return [
        {
            "id": 1,
            "title": "Example document 1",
            "content": f"Content related to {query}",
        },
        {
            "id": 2,
            "title": "Example document 2",
            "content": f"More information about {query}",
        },
    ][:limit]


def main():
    # Get settings
    settings = get_settings()
    print(f"Using model: {settings.openai_model}")

    # Define tools for the AI to use
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search the database for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    # Available tools mapping
    available_tools = {"search_database": search_database}

    # Initial conversation
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to a database.",
        },
        {"role": "user", "content": "Find information about machine learning"},
    ]

    # Call OpenAI with tool definitions
    response = call_with_tools(messages, tools)

    # Execute tool if called
    tool_result = execute_tool_call(response, available_tools)

    if tool_result:
        print(f"Tool called with result: {tool_result}")

        # First, add the assistant's message with tool_calls
        messages.append(
            {
                "role": "assistant",
                "content": None,  # Content is null when using tool_calls
                "tool_calls": response.choices[0].message.tool_calls,
            }
        )

        # Then add the tool result
        messages.append(
            {
                "role": "tool",
                "tool_call_id": response.choices[0].message.tool_calls[0].id,
                "name": response.choices[0].message.tool_calls[0].function.name,
                "content": str(tool_result),
            }
        )

        # Get final response
        final_response = call_with_tools(messages, tools, tool_choice="none")
        print(f"Final response: {final_response.choices[0].message.content}")
    else:
        print(f"Response: {response.choices[0].message.content}")


if __name__ == "__main__":
    main()
