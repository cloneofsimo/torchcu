import logging
import os
import sys

import torch

from judge import (
    load_function_signature,
    load_pytorch_function,
    prepare_inputs,
    run_pytorch_function,
)
from src.example_models import models
from src.generators.python_function import generate_functions
from src.generators.python_signature import generate_signature
from src.linter import is_valid_python

output_md_dir = "outputs/md"
output_py_dir = "outputs/py"
function_signature_divider = "\n# function_signature\n"

ITERATIONS = 1

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class Pipeline:
    def __init__(self, output_md_dir: str, output_py_dir: str):
        self.output_md_dir = output_md_dir
        self.output_py_dir = output_py_dir

    def generate_functions(self, reference_model) -> list[str]:
        return generate_functions(
            reference_model=reference_model,
            output_md_dir=self.output_md_dir,
            output_py_dir=self.output_py_dir,
        )

    def generate_input_signature(self, filepath: str, function_signature_divider: str):
        generate_signature(filepath)

    def delete_invalid_files(self, filepath: str):
        if not is_valid_python([filepath]):
            logger.warning(f"Invalid python code: {filepath}")
            os.remove(filepath)

    def run_pytorch_file(self, filepath: str) -> torch.Tensor:
        try:
            function_name, signature = load_function_signature(filepath)
            pytorch_func = load_pytorch_function(filepath, function_name)

            inputs = prepare_inputs(signature)
            output = run_pytorch_function(pytorch_func, inputs)

            logger.info(f"Output: {output.shape}, {output.dtype}")
            return output
        except Exception as e:
            logger.error(f"Failed to run PyTorch function: {e}")

    def get_output_signature(self, output: torch.Tensor):
        shape = f'({", ".join(output.shape)})'
        dtype = output.dtype
        return f"({shape}, {dtype})"

    def write_output_signature(self, filepath: str, output: torch.Tensor):
        with open(filepath, "r") as f:
            code = f.read()

        try:
            function, signature = code.split(function_signature_divider)
        except ValueError:
            logger.warning(f"Failed to read function signature: {filepath}")
            return

        output_signature = self.get_output_signature(output)
        signature.split('"outputs": [')[1].split("]")[0].strip()

    def run(self, reference_model: str):
        py_filepaths = self.generate_functions(reference_model)
        for filepath in py_filepaths:
            self.generate_input_signature(filepath, function_signature_divider)
            self.delete_invalid_files(filepath)
            output = self.run_pytorch_file(filepath)
            self.write_output_signature(filepath, output)


if __name__ == "__main__":
    pipeline = Pipeline(
        output_md_dir=output_md_dir,
        output_py_dir=output_py_dir,
    )

    i = 0
    for model in models.keys():
        i += 1
        logger.info(f"--- {i}/{len(models)}: Processing {model}")

        pipeline.run(model)

        if ITERATIONS is not None and i == ITERATIONS:
            break
