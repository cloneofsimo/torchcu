import logging
import os
import sys

import torch

from src.example_models import models
from src.generators.python_function import write_functions
from src.generators.python_signature import write_signature
from src.judge.python_run import run_pytorch_file
from src.linter import is_valid_python

output_md_dir = "outputs/md"
output_py_dir = "outputs/py"

ITERATIONS = 1

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class Pipeline:
    def __init__(self, output_md_dir: str, output_py_dir: str):
        self.output_md_dir = output_md_dir
        self.output_py_dir = output_py_dir

    def write_functions(self, reference_model) -> list[str]:
        return write_functions(
            reference_model=reference_model,
            output_md_dir=self.output_md_dir,
            output_py_dir=self.output_py_dir,
        )

    def write_input_signature(self, filepath: str):
        write_signature(filepath)

    def delete_invalid_files(self, filepath: str):
        if not is_valid_python([filepath]):
            logger.warning(f"Invalid python code: {filepath}")
            os.remove(filepath)

    def run_pytorch_file(self, filepath: str) -> torch.Tensor:
        try:
            output = run_pytorch_file(filepath)
            logger.info(f"Output: {output}")
            return output
        except Exception as e:
            logger.error(f"Failed to run PyTorch function: {e}")

    def write_output_signature(self, filepath: str, output: torch.Tensor):
        pass

    def run(self, reference_model: str):
        py_filepaths = self.write_functions(reference_model)
        for filepath in py_filepaths:
            self.write_input_signature(filepath)
            self.delete_invalid_files(filepath)
            output = self.run_pytorch_file(filepath)


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
