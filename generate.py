import logging
import sys

from generator import GeneratorPipeline
from generator.example_models import models

output_md_dir = "outputs/md"
output_py_dir = "outputs/py"
function_signature_divider = "\n# function_signature\n"

ITERATIONS = 1

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":
    pipeline = GeneratorPipeline(
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
