import os


def test_examples_exist():
    # The example report is kept, but generated images are optional and may be removed.
    assert os.path.exists("examples/report_example.md")
    # if images exist, ensure the folder is present (but images are optional)
    assert os.path.isdir("examples/images")


def test_examples_content():
    with open("examples/report_example.md", "r", encoding="utf-8") as f:
        txt = f.read()
    assert "Graph (original)" in txt
    assert "Benchmark" in txt
