# split up dataset into *.py and *.cu

import glob
import os
import shutil

def splitup(input_dir, output_dir):
    # remove output_dir if exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for file in glob.glob(f"{input_dir}/*.txt"):
        with open(file, "r") as f:
            filename = file.split("/")[-1]
            content = f.read()
            # find codeblock ```c++ and ```py
            try:
                py_code = content.split("```python")[1].split("```")[0]
                cu_code = content.split("```c++")[1].split("```")[0]
                # check if there is cutlass or cudnn in cu_code
                if "cutlass" in cu_code or "cudnn" in cu_code:
                    print(f"skip {file}")
                    continue
                
            except:
                print(f"Error: {file}")
                continue
            with open(f"{output_dir}/{filename.replace('.txt', '.py')}", "w") as f:
                f.write(py_code)
            with open(f"{output_dir}/{filename.replace('.txt', '.cu')}", "w") as f:
                f.write(cu_code)

if __name__ == "__main__":
    splitup("/home/ubuntu/cudamode/dataset_v2", "/home/ubuntu/cudamode/gfv2")