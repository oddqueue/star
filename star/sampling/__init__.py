__all__ = [
    "STARSampler",
    "SamplingResult",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .api import STARSampler, SamplingResult

    return {"STARSampler": STARSampler, "SamplingResult": SamplingResult}[name]
