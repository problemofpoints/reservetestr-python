# reservetestr-python

Python port of the [`reservetestr`](https://github.com/problemofpoints/reservetestr) R package for testing stochastic loss reserving methods against the CAS Loss Reserve Database (CLRD). The project builds on [`chainladder-python`](https://github.com/casact/chainladder-python) and provides:

- utilities to pull the CLRD triangles distributed with `chainladder` and isolate the Glenn Meyers testing subset;
- helpers to build train/test triangle pairs (upper-left vs. holdout) with an adjustable valuation year;
- wrappers around key stochastic reserving methods (`MackChainladder` and `BootstrapODPSample`) that return tidy back-testing metrics; and
- exhibit functions plus example notebooks that mirror the PP plot and histogram tooling from the original package.

## Installation

```bash
python -m pip install -e .
```

The `pyproject.toml` lists the core dependencies (`chainladder`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`). Install the optional `dev` extras to work with the notebooks:

```bash
python -m pip install -e .[dev]
```

## Usage

```python
import reservetestr as rt

# Build train/test triangles using the Meyers dataset exported from the R package.
records = rt.build_triangle_records()

# Run a Mack back-test on the Meyers subset for paid losses.
mack_results = rt.run_single_backtest(
    records,
    rt.testr_mack_chainladder,
    lines_to_include=["comauto", "ppauto", "wkcomp", "othliab"],
    loss_type="paid",
    method_label="mack_paid"
)

# Run the bootstrap ODP sampler with 1,000 simulations.
bootstrap_results = rt.run_single_backtest(
    records,
    rt.testr_bootstrap_odp,
    loss_type="paid",
    method_label="bootstrap_paid",
    n_sims=1000,
    hat_adj=False,
    random_state=22
)

# Create diagnostics similar to the R package exhibits.
fig = rt.create_pp_plot(mack_results, cv_limits=(0.0, 0.8), by_line=True)
fig.savefig("mack_pp_plot.png", dpi=200, bbox_inches="tight")
```

The repository also ships `meyers_triangles_long.csv`, exported directly from the R package. The dev-lag 10 column supplies the observed ultimate losses for each line/group combination, so the implied percentiles produced in Python align with the original reservetestr results.

### Choosing the valuation year

By default, `build_triangle_records()` uses the exact train/test split shipped with the R package (upper-left vs. lower-right triangles from the Meyers papers), so no calendar-year cut is required. If you want to experiment with alternative diagonals using the CLRD sample bundled with `chainladder-python`, pass a different `evaluation_year`:

```python
records = rt.build_triangle_records(evaluation_year=1992)
```

Setting `evaluation_year` to something earlier than 1997 ensures a non-empty testing triangle when working directly from the CLRD sample.

### Output schema

`run_single_backtest` returns a pandas `DataFrame` with the following columns (mirroring the R package):

| Column | Description |
| --- | --- |
| `line`, `group_id`, `company` | Triangle identifiers |
| `method` | Label passed to `run_single_backtest` |
| `actual_ultimate`, `actual_unpaid` | Observed totals derived from the holdout triangle |
| `mean_ultimate_est`, `mean_unpaid_est` | Model expectations |
| `stddev_est`, `cv_unpaid_est` | Dispersion metrics |
| `implied_pctl` | Implied percentile of the actual result under the modeled distribution |

## Notebooks

Two notebooks in `notebooks/` recreate the workflow from the R package:

1. `mack_chainladder_backtest.ipynb` – loads the Meyers subset, runs the Mack method, and plots PP charts/histograms.
2. `bootstrap_odp_backtest.ipynb` – repeats the workflow for `BootstrapODPSample`, highlighting how the simulated ultimates compare with observed outcomes.

Each notebook keeps execution lightweight (subsetting to a handful of triangles by default) and documents how to adjust the valuation year and simulation settings.

## Creating Custom Test Methods

The package provides a **flexible testing framework** so you don't need to write method-specific test functions. The framework handles all the plumbing (triangle extraction, actual ultimate calculation, derived metrics) - you just provide your reserving logic.

### Three Ways to Create Test Methods

#### 1. **Simple Decorator** (Easiest)

For methods that fit a model and return (mean, stddev), use the `@make_test_method` decorator:

```python
from reservetestr import make_test_method
import chainladder as cl

@make_test_method(distribution='lognormal')
def test_my_method(triangle, alpha=1.0, **kwargs):
    """My custom reserving method."""
    model = MyReservingModel(alpha=alpha).fit(triangle)
    mean_ult = float(model.ultimate_.sum().values)
    stddev = float(model.std_err_.sum().values)
    return mean_ult, stddev

# Use it directly with run_single_backtest
results = rt.run_single_backtest(
    records,
    test_my_method,
    method_label="my_method",
    alpha=1.5
)
```

#### 2. **Factory Function** (More Control)

For more complex cases, use `create_test_method()` to separate fitting and extraction:

```python
from reservetestr import create_test_method
import chainladder as cl

def fit_my_model(triangle, param1=10, param2='option', **kwargs):
    """Fit the model to a triangle."""
    return MyComplexModel(param1=param1, param2=param2).fit(triangle)

def extract_my_estimates(model, triangle):
    """Extract estimates from the fitted model."""
    mean_ultimate = model.get_ultimate_mean()
    stddev = model.get_ultimate_stddev()
    return mean_ultimate, stddev

# Create the test method
test_my_complex_method = create_test_method(
    fit_func=fit_my_model,
    extract_func=extract_my_estimates,
    distribution='lognormal'  # or 'normal'
)

# Use it
results = rt.run_single_backtest(
    records,
    test_my_complex_method,
    method_label="complex_method",
    param1=20,
    param2='advanced'
)
```

#### 3. **Bootstrap/Simulation Methods**

For methods that generate samples (bootstrap, MCMC, etc.), use `create_bootstrap_test_method()`:

```python
from reservetestr import create_bootstrap_test_method
import chainladder as cl
import numpy as np

def generate_my_samples(triangle, n_sims=1000, **kwargs):
    """Generate simulated ultimate values."""
    # Your simulation logic here
    samples = []
    for _ in range(n_sims):
        simulated_ult = simulate_one_ultimate(triangle, **kwargs)
        samples.append(simulated_ult)
    return np.array(samples)

# Create the test method
test_my_bootstrap = create_bootstrap_test_method(
    sample_func=generate_my_samples
)

# Use it
results = rt.run_single_backtest(
    records,
    test_my_bootstrap,
    method_label="my_bootstrap",
    n_sims=5000
)
```

### Required Output Structure

All test methods must return a dictionary with these keys:

| Key | Type | Description |
| --- | --- | --- |
| `actual_ultimate` | float | Observed ultimate loss from test triangle or actual_ultimates |
| `actual_unpaid` | float | actual_ultimate - latest_observed |
| `mean_ultimate_est` | float | Model's estimated ultimate loss |
| `mean_unpaid_est` | float | mean_ultimate_est - latest_observed |
| `stddev_est` | float | Standard deviation of ultimate estimate |
| `cv_unpaid_est` | float | Coefficient of variation of unpaid estimate |
| `implied_pctl` | float | Implied percentile of actual under model distribution |

**The framework handles all of this automatically** - you just provide your model's mean and standard deviation!

### Input Structure

Your test method will receive:

- `train_triangles`: Dictionary mapping loss types ('paid', 'incurred') to Triangle objects
- `test_triangles`: Dictionary mapping loss types to holdout Triangle objects
- `loss_type`: String indicating which triangle to use (default: 'paid')
- `actual_ultimates`: Optional dict of known ultimate values by loss type
- `**kwargs`: Any additional parameters your method needs

The framework provides helper functions:
- `get_triangle(triangles, loss_type)` - Extract the right triangle
- `resolve_actual_ultimate(...)` - Get the actual ultimate value
- `calculate_derived_metrics(...)` - Compute all derived metrics

### Complete Example: BornhuetterFerguson Method

```python
from reservetestr import create_test_method
import chainladder as cl
import numpy as np

def fit_bf_model(triangle, apriori=1.0, **kwargs):
    """Fit Bornhuetter-Ferguson model."""
    return cl.BornhuetterFerguson(apriori=apriori).fit(triangle)

def extract_bf_estimates(model, triangle):
    """Extract BF estimates (without analytical stddev, use bootstrap)."""
    mean_ult = float(model.ultimate_.sum().values)
    # BF doesn't have analytical stddev, so use a simple approximation
    # or return NaN and use bootstrap separately
    stddev = float('nan')  # Or implement bootstrap variance
    return mean_ult, stddev

test_bf = create_test_method(
    fit_func=fit_bf_model,
    extract_func=extract_bf_estimates,
    distribution='lognormal'
)

# Run it
results = rt.run_single_backtest(
    records,
    test_bf,
    method_label="bornhuetter_ferguson",
    apriori=0.8
)
```

### Advanced: Custom Percentile Calculation

If your method has a custom distribution, provide a `percentile_func`:

```python
def my_percentile_func(actual, model, triangle):
    """Custom percentile calculation using the model's distribution."""
    # Your custom logic here
    return model.compute_percentile(actual)

test_custom = create_test_method(
    fit_func=fit_my_model,
    extract_func=extract_estimates,
    percentile_func=my_percentile_func
)
```

## Next steps

- Add pytest-based regression tests for the loaders and method wrappers
- Expand exhibit helpers (error metrics, multi-method comparisons)
- Create examples for additional methods (Cape Cod, Bornhuetter-Ferguson, etc.)
