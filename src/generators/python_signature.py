import logging

from src.llm import LLM, Input

prompt_write_signature = """Read a function, and write a function signature for that.
It should specify the function's name, shapes for the input and output tensors, and data types.

Contstraints:
1. It should be in the following format:
```python
function_signature = {{
    "name": "<function_name>",
    "inputs": [
        (<shape_tuple>, <dtype>),
        (<shape_tuple>, <dtype>)
    ],
    "outputs": [
        (<shape_tuple>, <dtype>),
    ]
}}
```

2. Each shape should be greater than or equal to 4.
(4, 4) is the minimum shape for the tensors.

For example, consider the following function:
```python
def linear_transformation_activation(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output = torch.matmul(input_bf16, weight_bf16.t())
    return torch.relu(output).to(torch.float32)
```
    
The function signature for the above function would be:
```python
function_signature = {{
    "name": "linear_transformation_activation",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}}
```

Now, write a signature for below function. Do not repeat the function, just the signature.
Again, it should specify the function's name, parameter's names, dimensions for the tensors, and data types.
```python
{function}
```
"""


logger = logging.getLogger()


def write_signature(filepath: str):
    logger.info(f"Generating function signature for {filepath}")

    i = Input()
    llm = LLM()

    with open(filepath, "r") as f:
        content = f.read()
        i.add(prompt_write_signature.format(function=content))

    generated = ""
    for response in llm.generate(i):
        generated += response

    try:
        signature = generated.split("```python")[1].split("```")[0].strip()
    except IndexError:
        logger.warning(f"Failed to generate function signature: {generated}")
        return

    with open(filepath, "a") as f:
        f.write("\n\n\n" + signature)

    logger.info(
        f"Generated function signature for {filepath} (token usage: {llm.usage})"
    )
