import modal

app = modal.App("big-benchmark")

# Volumes
db_volume = modal.Volume.from_name("stopwatch-db", create_if_missing=True)
results_volume = modal.Volume.from_name("stopwatch-results", create_if_missing=True)
