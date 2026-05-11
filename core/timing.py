import time
import contextlib
import functools

# 0 = off, 1 = milestone timers, 2 = timers + cProfile
LEVEL = 0

_COL_TAG   = 8   # width of tag field
_COL_LABEL = 32  # width of label field

_deferred: list = []  # lines queued by defer_print(), flushed after each @timer call


def defer_print(line):
    """Queue a line to be printed after the enclosing @timer output."""
    _deferred.append(line)


def _fmt(tag, label, elapsed, note=''):
    note_str = f'  ({note})' if note else ''
    return f"  [{tag:<{_COL_TAG}}] {label:<{_COL_LABEL}} {elapsed:>8.3f} s{note_str}"


@contextlib.contextmanager
def timed(label, tag='timer', note=''):
    """Context manager that prints elapsed time when LEVEL >= 1."""
    if LEVEL < 1:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(_fmt(tag, label, elapsed, note() if callable(note) else note))


def timer(label=None, tag='timer', label_fn=None):
    """Decorator that prints elapsed time when LEVEL >= 1.

    label_fn(*args, **kwargs) → str  lets the label be derived from call arguments.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if LEVEL < 1:
                return fn(*args, **kwargs)
            lbl = label_fn(*args, **kwargs) if label_fn else (label or fn.__qualname__)
            _deferred.clear()
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(_fmt(tag, lbl, elapsed))
            for line in _deferred:
                print(line)
            _deferred.clear()
            return result
        return wrapper
    return decorator


@contextlib.contextmanager
def profile(label='profile', top_n=20):
    """Context manager that runs cProfile when LEVEL >= 2 and prints top_n functions."""
    if LEVEL < 2:
        yield
        return
    import cProfile
    import pstats
    import io
    pr = cProfile.Profile()
    pr.enable()
    try:
        yield
    finally:
        pr.disable()
        buf = io.StringIO()
        ps = pstats.Stats(pr, stream=buf).sort_stats('cumulative')
        ps.print_stats(top_n)
        print(f"\n── cProfile: {label} ──")
        print(buf.getvalue())
