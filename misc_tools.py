# -*- coding: utf8 -*-

from __future__ import print_function

from collections import Sequence


def pretty_time(delta):
    """scales and formats a time delta according to it's magnitude"""
    if delta < 0.001:
        return '{} usec'.format(int(delta * 1000000))
    if delta < 1:
        return '{} msec'.format(int(delta * 1000))
    return '{:.2f} sec'.format(delta)


class TimeIt(object):
    """context manager utility class for measuring execution time"""

    def __init__(self, context, printer=None):
        self.context = context
        if printer is None:
            self.printer = print
        else:
            self.printer = printer
        self.ts = self.te = 0

    def __enter__(self):
        import time
        self.ts = time.time()

    @property
    def elapsed(self):
        return self.te - self.ts

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.te = time.time()
        self.printer("{}: {}".format(self.context, pretty_time(self.elapsed)))


def recursive_map(func, seq):
    """applies a func to all elements recursively"""
    for item in seq:
        if isinstance(item, Sequence):
            yield type(item)(recursive_map(func, item))
        else:
            yield func(item)


def flatten(*seqs):
    """flattens a sequence of sequences"""
    result = []
    for seq in seqs:
        result.extend(seq)
    return tuple(result)


def memoize(func):
    """Memoization decorator for functions with one argument"""
    class MemoDict(dict):
        def __missing__(self, key):
            ret = self[key] = func(key)
            return ret
    return MemoDict().__getitem__


def memoize_range(size):
    """Memoization decorator for functions with one argument that is
    a number in range [0, size-1]"""
    cache = [None] * size

    def decorator(func):
        def wrapper(val):
            result = cache[val]
            if result is None:
                result = cache[val] = func(val)
            return result
        return wrapper
    return decorator
