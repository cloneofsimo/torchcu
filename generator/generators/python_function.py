import logging
import os
import random

from generator.example_models import models
from generator.llm import LLM

prompt_write_functions = """Here is a pytorch model source code.
{examples}

Your goal is to make a random pytorch code with a single function.
Use the provided code as a reference to write a function.

Constraints:
1. The function takes torch tensors and returns a torch tensor or tuple of them.
The function must take torch tensors as parameters,
and return a torch tensor or tuple of torch tensors as output.
If you need any other kind of parameters, make it as a constant.

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
Just make sure the function is executable.

5. Not too simple, not too complex.
The function must not be too simple or too complex.
Use the provided code as a reference.
It should be complex enough to test the candidate's ability to write a function.

Make sure.
Only torch tensors parameters, self-contained, and executable.
"""


logger = logging.getLogger()


def generate_markdown(
    reference_model: str,
    output_md_dir: str,
    max_reference_files: int = 3,
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
    return md_path, context


def generate_functions(
    md_filepath: str,
    output_py_dir: str,
) -> list[str]:
    with open(md_filepath, "r") as f:
        md = f.read()

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

        reference_model = md_filepath.split("/")[-1].replace(".md", "")
        filepath = f"{output_py_dir}/{reference_model}_{name}.py"
        if os.path.exists(filepath):
            filepath = (
                f"{output_py_dir}/{reference_model}_{name}_{random.randint(0, 1000)}.py"
            )
        with open(filepath, "w") as f:
            f.write(code)
        filepaths.append(filepath)

        logger.info(f"Saved snippet to {filepath}")

    return filepaths
