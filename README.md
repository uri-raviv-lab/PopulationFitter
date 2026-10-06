# Population Fitter (Python)

A Python/PyQt5 rewrite of `PopulationGUI.exe` ("Population Fitter") -- a tool for displaying linear/nonlinear combinations of
scattering curves and fitting their coefficients ("populations") against
experimental data.

## Running

```
pip install -r requirements.txt
python main.py
```

Tested with the Anaconda distribution (Python 3.12, PyQt5, numpy, scipy,
matplotlib already included).

**Windows/Anaconda note:** on some setups, importing PyQt5/matplotlib before
scipy causes `ImportError: DLL load failed while importing _spropack`. This
is worked around in `main.py` by importing `numpy`/`scipy.optimize` first --
if you build your own entry point, keep that order.

## Standalone .exe (no Python install required)

`PopulationFitter.exe` is a self-contained, single-file build -- just
copy and run it, no Python or packages needed on the target machine.
Download it from the [Releases](https://github.com/ariel234156/PopulationFitter/releases)
page (it is not stored in the repository itself, since it exceeds GitHub's
100 MB file limit; a local build lands in `dist/`).

To rebuild it after making changes:

```
build_venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onefile --windowed --name PopulationFitter main.py
```

`build_venv/` is a plain **PyPI**-based virtual environment (created from
`C:\Program Files\Python39\python.exe`, i.e. Python 3.9 -- **not** Anaconda's
Python) used only for building the exe -- it is not needed to run the app,
only to rebuild it. Several things had to be fixed to get a working frozen
build; all matter if you recreate this from scratch:

1. **Build from PyPI wheels, not Anaconda's.** Anaconda's numpy/scipy link
   against MKL, and when PyInstaller flattens all DLLs into one onefile
   payload, MKL's DLLs collide with scipy's bundled OpenBLAS (same failure
   as the note above, but this time unrecoverable by import order, since
   PyInstaller's frozen loader controls it). Plain `pip install numpy scipy`
   wheels bundle their own uniquely-named DLLs and don't collide.
2. **Bootstrap the venv from a plain python.org-style install, not
   `anaconda3\python.exe`.** Even just using conda's `python.exe` to create
   an otherwise-clean PyPI-wheel venv (`python -m venv`) produced a frozen
   build that crashed on startup with `ImportError: DLL load failed while
   importing _ctypes` -- conda's Python doesn't ship a self-contained DLLs
   layout the way python.org's installer does, and PyInstaller's dependency
   scan didn't follow the redirect (this repo also lives on a mapped network
   drive, which made it worse). Using `C:\Program Files\Python39\python.exe`
   to create `build_venv` avoided the problem entirely.
3. **`anthropic`'s 1.x line requires Python >= 3.10**, which is why
   `build_venv` stays on 3.9 with `anthropic` pinned to its last 0.x release
   (0.125.0) rather than chasing a newer Python just for that one package --
   confirmed the 0.x client still accepts the `output_config={"effort": ...}`
   parameter the AI advisor uses, so nothing in `ai_advisor.py` had to change.
4. **`sys.stdout`/`sys.stderr` are `None` in a `--windowed` build.** Any
   stray `warnings.warn()` or `print()` anywhere in the app or a dependency
   then raises `AttributeError` on a `None` stream and kills the process
   instantly with no visible error. `main.py` redirects both to
   `PopulationFitter.log` (next to the exe) before anything else runs if
   they're `None`, so warnings land in that file instead of crashing the app.

## What it does

- **Panels** (left column): each panel is either a loaded data **File** (Q, I
  columns) or an **Expression** combining other panels by their name, e.g.
  `a*Curve1 + (1-a)*Curve2`. Toggle File/Expression per panel, pick a color,
  toggle visibility, rename, set a **Subunits** count (e.g. 1 for monomer, 4
  for tetramer -- used for mass fractions, see below), remove, or export a
  panel's resolved curve. Expression panels can reference other expression
  panels (nested).
- **Variables** (middle): every free parameter used in any expression (panel
  expressions or the Fit box's expression) is auto-detected and listed with
  Value/Min/Max/Fit(vary) -- edit directly.
- **Fit** (bottom-left):
  - **Signal to fit:** a dropdown of loaded series (target/signal) curves --
    the one thing the fit will be matched against. Selecting a row in the
    series table below updates it too (and vice versa).
  - **Expression:** a dropdown of Expression panels (editable, so you can
    also type any raw formula, e.g. `a*Curve1 + (1-a)*Curve2`) -- the model
    to fit.
  - **Variables to fit:** checkboxes for every free parameter the chosen
    expression reaches (including through nested Expression panels) -- tick
    which ones this fit should vary; unticked ones stay fixed at their
    current value. This mirrors (and stays in sync with) the Fit column in
    the Variables table above.
  - **Fit Selected Signal** fits the chosen signal and updates the shared
    Variables table. **Fit All (independently)** fits every loaded series
    row separately (each gets its own coefficients -- e.g. per-sample
    population fractions) without touching the shared Variables table. Both
    also compute each component's **population (molar) fraction** and, using
    its panel's subunit count, its **mass fraction** -- shown as extra
    `MassFrac:<name>` columns in the series table.
  - **AI Fit Advisor...** (optional) sends a compact summary of the fit
    (expression, parameter values/bounds, RMS/Chi2, a downsampled
    data/model/residual table) to Claude and shows its plain-language
    commentary -- fit quality, whether a parameter looks pinned at a bound,
    whether residuals show a pattern suggesting a missing component. Runs on
    a background thread so it can't freeze the UI. Fully optional: needs the
    `anthropic` package and an API key (`Options > AI Settings...`, or the
    `ANTHROPIC_API_KEY` environment variable); the rest of the app works
    identically without either, and nothing is ever sent automatically.
  - `Options > Use Log Fitting` fits in log(I) space; `Fit Iterations...`
    caps the optimizer's iterations.
- **Plot** (right): matplotlib canvas with log(Q)/log(I) toggles, pan/zoom/
  save via the standard matplotlib toolbar.
- **Export**: series table (TSV, includes fitted coefficients and mass
  fractions per sample), series curves (data + fitted model per sample), or
  the graph as an image. Curve exports default to a `.out` filename and
  start with `#`-prefixed comment lines recording the expression, the value
  of every parameter used, mass fractions (for a fitted model curve), and
  the full file path of every currently loaded signal -- so an exported file
  is self-describing about exactly what produced it.
- **Drag & drop** is location-aware: drop `.out`/`.dat`/`.txt`/`.chi` files
  onto **Curves && Expressions** to add them as reference-curve File panels,
  or onto **Fit** to load them as series (signal) data -- same as "Add
  Multiple Panels..." / "Load series..." respectively. Dropping a signal
  onto Fit also auto-fills the Expression dropdown from an existing
  Expression panel (if one exists and none is chosen yet) and selects the
  new signal, so you can hit "Fit Selected Signal" immediately.
- **Save Session... / Load Session...** (`File` menu) saves everything --
  every panel including its actual loaded curve data (not just the file
  path, so the session is still reproducible if the original files move or
  disappear), every variable's value/bounds/vary state, every loaded signal
  with its fit results (model curve, RMS/Chi2, population/mass fractions),
  and a few UI settings (log fitting, fit iterations, log axes, current
  expression/signal selection) -- to one `.pfsession` (JSON) file, so a
  whole fitting session can be reopened later exactly as it was left.

## Fitting engine

Most population-fitting expressions are *linear* in their free parameters
(e.g. `a*Curve1 + (1-a)*Curve2`). For that common case, the engine detects
linearity numerically (probes the model at zero and each unit parameter
direction, then verifies against random points) and solves it **exactly**
with one bounded linear least-squares call -- no initial guess, no iteration
count, no local-minima risk, effectively instant. This is the same class of
problem the original tool solved via Eigen's SVD/QR, just verified generically
rather than assumed.

When the expression is genuinely nonlinear in its parameters (or log-space
fitting is on, which makes even a linear model's residual nonlinear), it
falls back to a **global search** (differential evolution, when bounds are
finite and reasonably tight) polished by local refinement, plus several
random-start local fits -- far more robust against a bad initial guess or a
local minimum than a single `scipy.optimize.least_squares` run from one
starting point. The method actually used ("linear (exact closed-form
solve)", "global search + local refine", "multi-start local optimization")
is reported in the fit status message and saved with each fit result.

**Population and mass fractions:** for each curve panel referenced by the
fit expression, the engine numerically recovers its effective linear weight
in the model (perturb that curve's values, see how the model responds) --
in a well-formed expression like `a*Curve1 + (1-a)*Curve2`, this *is* the
fitted population/molar fraction, recovered without needing to know which
literal parameter multiplies which curve. Multiplying each by the panel's
subunit count and renormalizing gives the mass fraction.

## Fixes over the original

- **File loading never crashes the app.** Header/comment lines and
  unparsable rows are skipped; `1.#INF`/`1.#IND`-style tokens and other
  non-finite values are dropped with a warning instead of taking down the
  whole program. A file with too little data raises a clear, catchable
  error dialog.
- **Expressions are validated safely.** Typed expressions are parsed and
  whitelisted via Python's `ast` module (only arithmetic ops and a small
  math-function allowlist) before evaluation, with no access to Python
  builtins -- a malformed or malicious expression can't crash or otherwise
  compromise the app, it just produces a clear "Invalid expression" message.
- **Self-references and circular references are detected explicitly**
  (`Expression 'X' references itself.`) instead of recursing until a stack
  overflow.
- **The fitting engine** (see above) is more numerically robust than the
  original's hand-rolled solver -- exact where possible, globally-searched
  where not -- and correctly discovers fit parameters through nested
  expression panels.
- **A global exception hook** (`main.py`) shows a dialog for any unexpected
  error instead of freezing or silently dying, and most UI actions are
  wrapped so one bad panel/expression doesn't take down the whole session.
  The AI advisor call runs on a background thread for the same reason --
  a slow network call can't freeze the window.
- **No artificial "26 open files" limit** -- that was a GDI-handle
  workaround in the original WinForms app; it doesn't apply here.
- **A whole session can be saved and reopened** -- the original had no such
  concept; every fit had to be redone from scratch each time the app opened.

## Known simplifications vs. the original

- No drag-and-drop panel reordering (panels are added at the end; use
  Remove + Add Panel to reorder).
- No manual Q-range slider for restricting a fit to a subrange -- crop the
  input file instead, or ask for this to be added.
