import logging
import os

from src.example_models import models
from src.llm import LLM, Input

prompt_write_functions = """These are example pytorch code.
Now, write some pytorch functions.

Constraints:
1. Takes and returns torch tensors.
It should be a function that takes torch tensors and returns a torch tensor.

2. The function should be self-contained.
Make it self-contained and independent of other functions.
Every functions or classes should be defined in the code.
Do not import any external functions or classes, except torch and numpy.

3. Answer has a specific format.
The answer format should be like below:
### <summary of the function>
```python
import torch

def <function_name>(<parameter_name>: torch.Tensor) -> torch.Tensor:
    # Your code here
```

Example:
### Perform a simple linear transformation and activation using bfloat16.
```python
import torch

def torch_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output = torch.matmul(input_bf16, weight_bf16.t())
    return torch.relu(output).to(torch.float32)
```

Make sure the function takes and returns torch tensors and is self-contained.
"""


logger = logging.getLogger()


def write_functions(
    reference_model: str, output_md_dir: str, output_py_dir: str
) -> list[str]:
    md = ""
    md_path = f"{output_md_dir}/learn_from_{reference_model}.md"
    if os.path.exists(md_path):
        md = open(md_path, "r").read()
    else:
        i = Input()
        llm = LLM()

        modelfiles = models[reference_model]
        for filename, content in modelfiles.items():
            prompt = f"file: {filename}\n{content}"
            i.add(prompt)
        i.add(prompt_write_functions)

        logger.info(
            f"Generating functions based on {reference_model} ({len(modelfiles)} files)"
        )
        with open(md_path, "w") as f:
            for response in llm.generate(i):
                md += response
                f.write(response)

        logger.info(f"Generated markdown file: {md_path} (token usage: {llm.usage})")

    filepaths = []
    snippets = [s for s in md.split("###") if s != ""]
    logger.info(f"Generated {len(snippets)} snippets")
    for snippet in snippets:
        try:
            name = snippet.split("\n")[0].strip().replace(" ", "_").lower()
            code = snippet.split("```python")[1].split("```")[0].strip()
        except IndexError:
            logger.warning(f"Failed to parse snippet, model output: {snippet}")
            continue

        filepath = f"{output_py_dir}/{reference_model}_{name}.py"
        with open(filepath, "w") as f:
            f.write(code)
        filepaths.append(filepath)

        logger.info(f"Saved snippet to {filepath}")

    return filepaths
