from PIL import Image


class LazyLineDataset:
    def __init__(self, samples, format_fn, cache_size: int = 8):
        self.samples = samples
        self.format_fn = format_fn
        self._page_cache = {}
        self._cache_max = cache_size

    def _get_page_image(self, path):
        if path not in self._page_cache:
            if len(self._page_cache) >= self._cache_max:
                oldest = next(iter(self._page_cache))
                self._page_cache[oldest].close()
                del self._page_cache[oldest]
            self._page_cache[path] = Image.open(path).convert("RGB")
        return self._page_cache[path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        result = self.format_fn(self.samples[idx], self._get_page_image)
        if result is None:
            raise ValueError(f"Invalid sample at index {idx}")
        return result