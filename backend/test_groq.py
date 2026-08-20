from groq import Groq

api_key = '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'
client = Groq(api_key=api_key)

tools = [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather conditions for a specific city. Use this tool when the user asks about current weather, temperature, rain, or weather conditions in a location. Do not use this tool for historical weather or general questions about climate. If a country is mentioned, the city should be from that country.",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string",
            "description": "The city whose weather should be retrieved, for example 'Bengaluru' or 'New York'."
          },
          "country": {
            "type": "string",
            "description": "The country containing the city, for example 'India' or 'United States'."
          },
          "units": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature units requested by the user. Default to celsius if the user does not specify units."
          }
        },
        "required": ["city", "country", "units"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_order_status",
      "description": "Retrieve the current status of a customer's order using its order ID. Use this when the user asks where their order is, whether it has shipped, or when it is expected to arrive.",
      "parameters": {
        "type": "object",
        "properties": {
          "order_id": {
            "type": "string",
            "description": "Unique order identifier, for example 'ORD-10482'."
          }
        },
        "required": ["order_id"]
      }
    }
  }
]

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
      {
        "role": "system",
        "content": "Answer user's questions, use the correct tool only if needed"
      },
      {
        "role": "user",
        "content": "Has order ord-19910 been shipped?\n\n"
      },
    ],
    tools=tools,
    temperature=1,
    max_completion_tokens=2048,
    top_p=1,
    reasoning_effort="medium",
    stream=True,
    stop=None
)

# print("Streaming Text Content: ")
# for chunk in completion:
#     print(chunk.choices[0].delta.content or "", end="")

# print("Streaming Tool Arguments: ", end="", flush=True)

# for chunk in completion:
#     delta = chunk.choices[0].delta
    
#     # Check if the stream chunk contains a tool call fragment
#     if delta.tool_calls:
#         for tool_call in delta.tool_calls:
#             if tool_call.function and tool_call.function.arguments:
#                 # Print the JSON arguments fragment by fragment as they arrive
#                 print(tool_call.function.arguments, end="", flush=True)
                
#     elif delta.content:
#         # Fallback in case the model outputs standard text
#         print(delta.content, end="", flush=True)

# print("\n")

for chunk in completion:
    delta = chunk.choices[0].delta

    if delta.tool_calls:
        for tool_call in delta.tool_calls:

            if tool_call.function:
                if tool_call.function.name:
                    print("Tool:", tool_call.function.name)

                if tool_call.function.arguments:
                    print(
                        "Arguments fragment:",
                        tool_call.function.arguments
                    )