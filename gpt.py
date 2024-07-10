from openai import OpenAI
from dotenv import dotenv_values

config = dotenv_values(".env")

client = OpenAI(
    base_url=config.get("OPENAI_API_BASE"),
    api_key=config.get("OPENAI_API_KEY"),
)


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
stream = client.chat.completions.create(
  model="gpt-4",
  messages=messages,
  tools=tools,
  tool_choice="auto"
)
print(stream)
# stream = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     # model="gpt-4",
#     messages=[
#         {"role": "system", "content": "You are a children's book author."},
#         {"role": "user", "content": "List a format that need 10 sentences. Write a 100 words of the book about 3 pigs story. "},
#     ],
#     stream=True,
# )


# try:
#     for chunk in stream:
#         print(chunk)

#         if (
#             chunk.choices
#             and chunk.choices[0].delta
#             and chunk.choices[0].delta.content is not None
#         ):
#             print(chunk.choices[0].delta.content, end="")

# except IndexError:
#     print("Error: The API response did not contain the expected data structure.")
# except Exception as e:

#     print(f"An error occurred: {e}")
