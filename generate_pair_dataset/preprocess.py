# split up dataset into *.py and *.cu

import glob

def splitup(input_dir, output_dir):
    for file in glob.glob(f"{input_dir}/*.txt"):
        with open(file, "r") as f:
            filename = file.split("/")[-1]
            content = f.read()
            # find codeblock ```c++ and ```py
            py_code = content.split("```python")[1].split("```")[0]
            cu_code = content.split("```c++")[1].split("```")[0]
            with open(f"{output_dir}/{filename.replace('.txt', '.py')}", "w") as f:
                f.write(py_code)
            with open(f"{output_dir}/{filename.replace('.txt', '.cu')}", "w") as f:
                f.write(cu_code)

if __name__ == "__main__":
    splitup("/home/ubuntu/cudamode/dataset", "/home/ubuntu/cudamode/gf_dataset")