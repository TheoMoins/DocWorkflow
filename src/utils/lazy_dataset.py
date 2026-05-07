from PIL import Image


class LazyLineDataset:
    def __init__(self, samples, format_fn, cache_size: int = 8):
        self.samples = samples
        self.format_fn = format_fn
        self._cache_max = cache_size
        self._page_cache = {}  # keyed by path, value is a fully loaded copy

    def _get_page_image(self, path):
        if path not in self._page_cache:
            if len(self._page_cache) >= self._cache_max:
                oldest = next(iter(self._page_cache))
                del self._page_cache[oldest]  # no .close() — it's already a copy
            # .copy() decouples from the file handle entirely
            img = Image.open(path).convert("RGB")
            img.load()        # force full decode into memory
            self._page_cache[path] = img.copy()
            img.close()       # safe to close the file-backed original now
        return self._page_cache[path].copy()  # return a copy, never the cached ref

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        result = self.format_fn(self.samples[idx], self._get_page_image)
        if result is None:
            raise ValueError(f"Invalid sample at index {idx}")
        return result