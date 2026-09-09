# External researcher full-pipeline replication

## Goal
Follow the README CLI as an external researcher and re-run Path 2 (`mq reproduce full-pipeline`), fixing blockers discovered along the way.

## Outcome
- Path 2 full pipeline succeeded end-to-end after obtaining Alizadeh/Gonzalez external data.
- Hardened `mq papers download` against arXiv OpenSearch field-shape changes.
- Added retries for transient Zenodo HTTP failures during `--download-missing`.
