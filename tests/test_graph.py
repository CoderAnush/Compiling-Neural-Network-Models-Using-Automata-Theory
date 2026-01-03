import pytest
from model.train_mnist import get_model
from graph.extract_graph import extract_graph


def test_extract_graph_basic():
    """The graph extractor should return a dict with an 'Input' node."""
    model = get_model(pretrained=False)
    g = extract_graph(model)
    assert isinstance(g, dict)
    assert "Input" in g
    assert len(g) > 0
