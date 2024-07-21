import time
# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client1 = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


from huggingface_hub import InferenceClient
client2 = InferenceClient(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        token="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",
    )

start = time.process_time()
completion = client1.chat.completions.create(
  model="Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF",
  messages=[
    {"role": "system", "content": "name is Jarvis(reference:Iron-Man) Always answer in short."},
    {"role": "user", "content": "give me a recipie to cheeze tadka maggie at home"}
  ],
  temperature=0.7,
  stream = False,
)
print(completion.choices[0].message.content)
# your code here    
print("-",time.process_time() - start,"sec")
start = time.process_time()
message = client2.chat_completion(
    messages=[
    {"role": "system", "content": "name is Jarvis(reference:Iron-Man) Always answer in short."},
    {"role": "user", "content": "give me a recipie to cheeze tadka maggie at home"}
  ],
    temperature=0.7,
    stream=False,
)
print(completion.choices[0].message.content)
# your code here    
print("-",time.process_time() - start,"sec")