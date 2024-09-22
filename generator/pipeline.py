import logging
import os

import torch

from generator.generators.python_function import generate_functions
from generator.generators.python_signature import (
    function_signature_divider,
    generate_signature,
)
from generator.linter import is_valid_python
from judge import (
    load_function_signature,
    load_pytorch_function,
    prepare_inputs,
    run_pytorch_function,
)

logger = logging.getLogger()


class GeneratorPipeline:
    def __init__(self, output_md_dir: str, output_py_dir: str):
        self.output_md_dir = output_md_dir
        self.output_py_dir = output_py_dir

    def generate_functions(self, reference_model) -> list[str]:
        return generate_functions(
            reference_model=reference_model,
            output_md_dir=self.output_md_dir,
            output_py_dir=self.output_py_dir,
        )

    def generate_input_signature(self, filepath: str):
        generate_signature(filepath)

    def delete_invalid_files(self, filepath: str):
        if not is_valid_python([filepath]):
            logger.warning(f"Invalid python code: {filepath}")
            os.remove(filepath)

    def run_pytorch_file(self, filepath: str) -> torch.Tensor | tuple[torch.Tensor]:
        function_name, signature = load_function_signature(filepath)
        pytorch_func = load_pytorch_function(filepath, function_name)

        inputs = prepare_inputs(signature)
        output = run_pytorch_function(pytorch_func, inputs)

        logger.info(f"Output: {output.shape}, {output.dtype}")
        return output

    def get_output_signature(self, output: torch.Tensor | tuple[torch.Tensor]):
        def get_signature(tensor: torch.Tensor):
            shape = f'({", ".join(tensor.shape)})'
            dtype = tensor.dtype
            return f"({shape}, {dtype})"

        if isinstance(output, tuple):
            return ",\n".join([get_signature(tensor) for tensor in output])
        elif isinstance(output, torch.Tensor):
            return get_signature(output)
        else:
            raise ValueError(f"Invalid output type: {type(output)}")

    def write_output_signature(
        self, filepath: str, output: torch.Tensor | tuple[torch.Tensor]
    ):
        with open(filepath, "r") as f:
            code = f.read()

        try:
            function, signature = code.split(function_signature_divider)
        except ValueError:
            raise ValueError(f"Failed to read function signature: {filepath}")

        output_signature = self.get_output_signature(output)
        old_front = signature.split('"outputs": [')[0]
        old_back = signature.split('"outputs": [')[1].split("]")[1]
        new_signature = old_front + f'"outputs": [{output_signature}]' + old_back

        with open(filepath, "w") as f:
            f.write(function + function_signature_divider + new_signature)

    def run(self, reference_model: str):
        py_filepaths = self.generate_functions(reference_model)
        for filepath in py_filepaths:
            self.generate_input_signature(filepath)
            self.delete_invalid_files(filepath)
            output = self.run_pytorch_file(filepath)
            self.write_output_signature(filepath, output)
