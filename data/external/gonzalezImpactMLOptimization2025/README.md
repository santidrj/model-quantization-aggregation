Public replication data is fetched automatically on first read when missing
(`final_ds_image-classification.csv` from the Zenodo archive). That first fetch
downloads an ~853 MB archive over the network, extracts only the required CSV,
and discards the archive. The data files are not committed to this repository.

Manual fallback: [Zenodo record 14845545](https://doi.org/10.5281/zenodo.14845545).
