# registry.py
"""
Generic id -> function registry.

CLAUDE.md describes a "registry + recipe" extension convention, but until now the
only thing implementing it was gui/prediction_ranges.py's private _METHODS dict.
This generalizes that pattern so metrics, pools and comparators all extend the
same way: write a function, decorate it, done -- no caller edits.
"""


class Registry:
    def __init__(self, kind):
        self.kind = kind
        self._fns = {}
        self._labels = {}
        self._caveats = {}
        self._meta = {}

    def register(self, id, label=None, caveat=None, **meta):
        def decorator(fn):
            if id in self._fns:
                raise ValueError(f"{self.kind} {id!r} is already registered")
            self._fns[id] = fn
            self._labels[id] = label or id
            self._caveats[id] = caveat
            self._meta[id] = meta
            return fn
        return decorator

    def get(self, id):
        if id not in self._fns:
            raise KeyError(
                f"Unknown {self.kind} {id!r}. Available: {self.available()}"
            )
        return self._fns[id]

    def available(self):
        return list(self._fns)

    def label(self, id):
        return self._labels.get(id, id)

    def caveat(self, id):
        return self._caveats.get(id)

    def meta(self, id, key, default=None):
        return self._meta.get(id, {}).get(key, default)

    def __contains__(self, id):
        return id in self._fns

    def __len__(self):
        return len(self._fns)
