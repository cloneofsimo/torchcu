from flake8.api import legacy as flake8

# https://pycodestyle.pycqa.org/en/latest/intro.html#error-codes
# https://flake8.pycqa.org/en/latest/user/error-codes.html
ignore = [
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "E7",
    "E8",  # ignore all but potential runtime errors
    "F401",  # ignore unused imports
    "W",  # ignore all warnings
]


def is_valid_python(file_paths: list[str]):
    style_guide = flake8.get_style_guide(ignore=ignore)
    report = style_guide.check_files(file_paths)
    return report.get_statistics("E") == [] and report.get_statistics("F") == []
