# Azure AI Foundry Data Analyst

Runs an Azure AI Foundry agent with the Code Interpreter tool over one or more CSV files, then uploads any generated charts to Azure Blob Storage.

## Setup

1. Install dependencies:
   ```
   uv sync
   ```
2. Fill in `.env`:
   ```
   PROJECT_ENDPOINT=
   MODEL_DEPLOYMENT_NAME=
   BLOB_STORAGE_CONNECTION_STRING=
   BLOB_STORAGE_CONTAINER_NAME=
   ```
3. Authenticate with Azure:
   ```
   az login
   ```

## Run

```
uv run main.py --prompt "<your prompt>" --csv path/to/file1.csv [path/to/file2.csv ...]
```

Example:
```
uv run main.py --prompt "Create a bar chart of personnel types across all years" --csv sample_dataset.csv
```
