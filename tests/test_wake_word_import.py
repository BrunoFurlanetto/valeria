import importlib


def test_wake_word_module_imports_without_starting_demo():
    module = importlib.import_module("app.network.exec")

    assert hasattr(module, "WakeWordEngine")
