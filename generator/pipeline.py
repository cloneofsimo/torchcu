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

    def run_pytorch_file(self, filepath: str) -> torch.Tensor | tuple[torch.Tensor]:
        function_name, inputs_signature, _ = load_function_signature(filepath)
        pytorch_func = load_pytorch_function(filepath, function_name)

        inputs = prepare_inputs(inputs_signature)
        output = run_pytorch_function(pytorch_func, inputs)

        return output

    def validate_output(self, output: torch.Tensor | tuple[torch.Tensor]):
        def validate_tensor(tensor: torch.Tensor):
            if len(tensor.shape) == 0:
                raise ValueError("We don't need scalar outputs for now.")

        if isinstance(output, tuple):
            for tensor in output:
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Invalid output type: {type(tensor)}")
                validate_tensor(tensor)
        elif isinstance(output, torch.Tensor):
            validate_tensor(output)
        else:
            raise ValueError(f"Invalid output type: {type(output)}")

    def get_output_signature(self, output: torch.Tensor | tuple[torch.Tensor]):
        def get_signature(tensor: torch.Tensor):
            shape = f'({", ".join([str(s) for s in tensor.shape])})'
            dtype = tensor.dtype
            return f"({shape}, {dtype})"

        if isinstance(output, tuple):
            return (
                "\n        "
                + ",\n        ".join([get_signature(tensor) for tensor in output])
                + "\n    "
            )
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

            if not is_valid_python([filepath]):
                logger.warning(f"Invalid python code: {filepath}")
                os.remove(filepath)
                continue

            try:
                output = self.run_pytorch_file(filepath)
            except Exception as e:
                logger.warning(
                    f"Failed to run PyTorch function: {filepath}\nRemoving file."
                )
                logger.exception(e)
                os.remove(filepath)
                continue

            try:
                self.validate_output(output)
            except Exception as e:
                logger.warning(
                    f"Invalid output from PyTorch function: {filepath}\nError: {e}\nRemoving file.\n"
                )
                os.remove(filepath)
                continue

            self.write_output_signature(filepath, output)
