# TensorMask

Tracks which points of a large multi-dimensional parameter space have been marked, compressing
whole axis-aligned "hyper-rows" into lower-dimensional zones as they fill up -- so a marked region
that lines up with the axes of the space costs far less memory than the size of the space itself,
even for spaces far larger than could be stored point-by-point (a highly irregular region that
doesn't align with any axis degrades toward one entry per marked point).

```cpp
#include "TensorMask.hpp"

TensorMask<> mask({1080, 1920, 3}); // e.g. a video frame: row, column, channel
mask.add(mask.flatten({0, 0, 0}));  // mark one point...
mask.add(mask.flatten({0, 0, 1}));
mask.add(mask.flatten({0, 0, 2}));  // ...every channel of pixel (0, 0) is now marked

mask.contains(mask.flatten({0, 0, 1})); // true
mask.contains(mask.flatten({0, 1, 0})); // false -- a different pixel
```

## Why not just a `std::vector<bool>` or `std::unordered_set`?

Both work fine until the space is too large to enumerate point-by-point -- a `std::vector<bool>`
needs one bit per point up front regardless of how many are actually marked, and a flat set needs
one entry per marked point. `TensorMask` instead notices when marking a point completes an entire
row along one dimension (every channel of a pixel, every pixel of a row, ...) and rolls that
information up into a zone over the *remaining* dimensions -- so a whole marked plane, row, or
sub-cube costs one entry in a lower-dimensional zone rather than one entry per point it covers.
This trades some per-`contains()`-call work (checking a handful of zones instead of one flat
lookup) for a memory bound that tracks how *regular* the marked region is, rather than how large
the space is.

## Views: `configure()`

A `TensorMask` is constructed with the full shape of the space, but `add()`/`contains()`/`index()`/
`flatten()`/`size()` all operate against whichever subset of dimensions is currently
"configured" -- by default, every dimension. `configure()` narrows that to a subset:

```cpp
TensorMask<> mask({2, 3, 4});
mask.configure({0, 1});     // work in just the first two dimensions
mask.add(mask.flatten({0, 0}));
mask.configure({0, 1, 2});  // back to the full space
mask.contains(mask.flatten({0, 0, 0})); // true -- marks from any past configuration persist
mask.contains(mask.flatten({0, 0, 3})); // also true -- the whole (0, 0, *) row was marked
```

Switching configuration never discards anything -- it only changes which dimensions the public
methods address.

## Ordering requirement

Within whichever shape is currently configured, flat indices must be `add()`ed in ascending order.
This is what lets `TensorMask` notice a completed row the moment its last point is added, without
re-scanning anything. Violating it is checked via `ensure()` (see [Requirements](#requirements)
below), which aborts the process rather than returning an error -- this is meant to catch a bug in
the caller supplying indices, not to be a recoverable, expected condition.

## Requirements

- C++20 or later
- Header-only -- copy the header into your project and `#include` it
- Requires [`Flattener.hpp`](https://github.com/lrmoorejr/flattener) on your include path --
  vendored here as a git submodule (`flattener/`); clone with `--recurse-submodules` or run
  `git submodule update --init` after cloning
- Optional: [`Ensure.hpp`](https://github.com/lrmoorejr/ensure) for a formatted diagnostic if the
  ascending-order requirement (or an internal invariant) is ever violated; falls back to plain
  `assert()` if not present -- either way, this is only active outside `NDEBUG` builds

## License

Apache License 2.0 -- see [LICENSE](LICENSE).
