# Run Download Phase Only (3 downloaders, queries from YAML file)


## Search-queries YAML file format

```yaml
# search_queries.yaml
queries:
  - "glass door installation tutorial"
  - "sliding glass door"
  - "..."
```

## Config file (download-only, 3 downloaders)

Create `download_only.yaml`:

```yaml
project_name: "my_project"
output_dir: "./output"

input:
  search_enabled: true
  search_queries_file: "search_queries.yaml"   # path to your queries YAML
  max_results_per_query: 50

supervisor:
  download_workers: 3   # 3 parallel downloaders
  extract_workers: 0    # disable
  filter_workers: 0     # disable
  dedup_workers: 0      # disable
  detect_workers: 0     # disable
```

## Commands to run

```bash
# 1. Initialize the database (first time only)
data-miner init-db

# 2. Populate DB with videos found from search queries
data-miner populate --config download_only.yaml

# 3. Generate supervisor config with the chosen worker counts
data-miner workers setup --config download_only.yaml

# 4. Start the workers (only the 3 download workers will run)
data-miner workers start

# 5. Monitor progress
data-miner status --project my_project
data-miner workers status

# 6. When downloads finish, stop the workers
data-miner workers stop
```

## Verification

- `data-miner status --project my_project` should show `Stage: POPULATING` and videos progressing from `PENDING` → `DOWNLOADED` (downloads complete) and remain there since no extractor runs.
- `data-miner workers status` should list only download workers (3 of them) as RUNNING.
- Downloaded videos land in `./output/videos/` ([data_miner/config/default.yaml:89](data_miner/config/default.yaml#L89)).
