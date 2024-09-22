import logging
import os
import random

from generator.example_models import models
from generator.llm import LLM

prompt_write_functions = """Here is a pytorch model source code.
{examples}

Your goal is to make a simular but simpler model with a single function.

Constraints:
1. The function takes torch tensors and returns a torch tensor or tuple of them.
The function must take torch tensors as parameters,
and return a torch tensor or tuple of torch tensors as output.

2. The function must be self-contained.
Make it self-contained and independent of other functions.
Every functions or classes must be defined in the code.
Do not import any external functions or classes, except fo torch, numpy and math.

3. Answer has a specific format.
The answer must be formatted like below:
### <summary of the function>
```python
import torch

def <function_name>(<parameter_name>: torch.Tensor, <parameter_name>: torch.Tensor, ...) -> torch.Tensor:
    # Your code here
```

4. It must be executable.
Even if a function may not run as expected, it must be able to run without any errors.
If provided model code is not enough to implement the function, you can make assumptions.
Just make sure the function is executable.

Make sure the function takes and returns torch tensors and is self-contained.
"""


logger = logging.getLogger()


def generate_functions(
    reference_model: str,
    output_md_dir: str,
    output_py_dir: str,
    max_reference_files: int = None,
    reuse_existing_md: bool = False,
) -> tuple[list[str], list[dict[str, str]]]:
    context = []
    md = ""
    md_path = f"{output_md_dir}/{reference_model}.md"
    if reuse_existing_md and os.path.exists(md_path):
        md = open(md_path, "r").read()
    else:
        llm = LLM()

        examples = ""
        modelfiles = models[reference_model]
        count = 0
        for filename, content in modelfiles.items():
            if max_reference_files is not None and count >= max_reference_files:
                break
            examples += f"filename: {filename}\n{content}\n\n"
            count += 1

        context.append(
            {
                "role": "user",
                "content": prompt_write_functions.format(examples=examples),
            }
        )

        logger.info(
            f"Generating functions based on {reference_model} ({len(modelfiles)} files)"
        )
        with open(md_path, "w") as f:
            for response in llm.generate(
                context,
                max_tokens=1024 * 20,
            ):
                md += response
                f.write(response)
        context.append({"role": "assistant", "content": md})

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
        if os.path.exists(filepath):
            filepath = (
                f"{output_py_dir}/{reference_model}_{name}_{random.randint(0, 1000)}.py"
            )
        with open(filepath, "w") as f:
            f.write(code)
        filepaths.append(filepath)

        logger.info(f"Saved snippet to {filepath}")

    return filepaths, context
