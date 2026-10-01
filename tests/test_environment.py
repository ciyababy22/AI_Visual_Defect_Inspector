import sys


def test_python_version():
    assert sys.version_info >= (3, 13)


def test_python_environment():
    print(f"Python version: {sys.version}")