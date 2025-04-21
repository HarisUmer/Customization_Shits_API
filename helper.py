import lamini
import json
lamini.api_key ="Enter You Lamini Api Key"
chat = lamini.Lamini("meta-llama/Meta-Llama-3-8B-Instruct")

def generate_response(prompt: str) -> str:
    """
    Generate a response using Lamini's API.
    """
    response = chat.generate(prompt)
    return response.strip()

def catterJson(response: str):
   prompt = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
Take any JSON input provided by the user, even if the JSON is faulty or improperly formatted. Your task is to fix the structure and formatting to produce a valid JSON object or array. Do not include any additional explanations or text other than the JSON response.

<|Example|>
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{ "name": "Ahmed", "age": 25, "address": { "city": "New York", "state": "NY", "zipcode": "10001" }
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{
    "name": "Ahmed",
    "age": 25,
    "address": "
        "city": "New York",
        "state": "NY",
        "zipcode": "10001"
    "
}

<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{ "key": "value" "another_key": "another_value" }
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{
    "key": "value",
    "another_key": "another_value"
}

<|start_Task|>
Now apply this logic to the given JSON input. Response:
- Output the fixed JSON as the response.
"""
   t1 = f"""
<|start_Input|>
{response}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
   prompt = prompt + t1
   dir  = generate_response(prompt=prompt)
   print(dir)
   try:
      return json.loads(dir)
   except:
      catterJson(dir)
