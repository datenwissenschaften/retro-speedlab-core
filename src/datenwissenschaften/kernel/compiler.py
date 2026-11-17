import sys
from collections.abc import Callable
from types import FunctionType, ModuleType
from typing import Any

Jit = Callable[[FunctionType], Any]


class Compiler:
    def __init__(self, package: str, jit: Jit) -> None:
        self.package = package
        self.jit = jit
        self.namespaces: dict[str, dict[str, Any]] = {}
        self.functions: dict[FunctionType, Any] = {}

    def owns(self, value: object) -> bool:
        if not isinstance(value, FunctionType):
            return False
        return value.__module__ == self.package or value.__module__.startswith(f"{self.package}.")

    def namespace(self, module: str) -> dict[str, Any]:
        if module in self.namespaces:
            return self.namespaces[module]
        names = self.namespaces[module] = dict(vars(sys.modules[module]))
        for name, value in list(names.items()):
            if self.owns(value):
                names[name] = self.function(value)
        return names

    def function(self, original: FunctionType) -> Any:
        names = self.namespace(original.__module__)
        if original.__closure__ is not None:
            raise TypeError(f"A kernel function cannot be a closure: {original.__qualname__}")
        if original not in self.functions:
            self.functions[original] = self.jit(FunctionType(original.__code__, names, original.__name__))
        return self.functions[original]


def compiled(kernel: ModuleType, jit: Jit) -> dict[str, Any]:
    return Compiler(kernel.__name__.partition(".")[0], jit).namespace(kernel.__name__)
