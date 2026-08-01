# Evidence runtime directory marker

The frozen `fig171819_confirmed_compare.py` import chain imports
`_v2_repro_nc12.py`, whose module initialization calls `os.makedirs()` for
this directory with `exist_ok=True`.

The Fig. 17/18/19 evidence launcher includes this marker in its read-only
runtime source closure so the directory exists before project imports.  The
scientific attribution process does not read or write the historical
`repro_nc12` caches.
