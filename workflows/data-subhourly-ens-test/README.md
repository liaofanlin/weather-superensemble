# data-subhourly-ens-test

Four-member time-lagged HRRR subhourly ensemble over Boulder.

For every 15-minute valid time from **+00:15 through +03:00** relative to the
base cycle, the workflow creates one **2 x 2 panel figure**.

For base cycle `2026-08-26 23Z`, the figure valid at `2026-08-26 23:15 UTC`
contains:

| Member | Init | Lead |
|---|---|---|
| 1 | 08/26 23Z | +00:15 |
| 2 | 08/26 22Z | +01:15 |
| 3 | 08/26 21Z | +02:15 |
| 4 | 08/26 20Z | +03:15 |

The same logic is used for every subsequent valid time.

For example, valid `08/26 23:30 UTC` uses:

- 23Z +00:30
- 22Z +01:30
- 21Z +02:30
- 20Z +03:30

The final valid time is base +03:00. Therefore, the oldest member needs a
+06:00 HRRR subhourly forecast.

## Files

- `run_hrrr_subhourly_ens_test_workflow.py`
- `fetch_hrrr_subhourly_ens_test.py`
- `regrid_hrrr_subhourly_ens_test.py`
- `plot_hrrr_subhourly_ens_test.py`

## Run

Default example:

```bash
python run_hrrr_subhourly_ens_test_workflow.py
```

Or:

```bash
python run_hrrr_subhourly_ens_test_workflow.py --date 20260826 --cycle 23
```

Example figure names:

```text
subhourly_ens_valid_20260826_2315UTC.png
subhourly_ens_valid_20260826_2330UTC.png
...
subhourly_ens_valid_20260827_0200UTC.png
```
