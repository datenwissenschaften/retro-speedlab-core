from importlib.util import find_spec

collect_ignore = [] if find_spec("torch") else ["test_policy.py"]
