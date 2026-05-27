import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AutoCodeInterpreterToolParam,
    CodeInterpreterTool,
    PromptAgentDefinition,
)
from azure.identity import DefaultAzureCredential

from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run an Azure AI Foundry data-analyst agent over one or more CSV files."
    )
    parser.add_argument("--prompt", required=True, help="The user prompt to send to the agent.")
    parser.add_argument(
        "--csv",
        required=True,
        nargs="+",
        help="One or more CSV file paths to upload for analysis.",
    )
    args = parser.parse_args()

    project_endpoint = os.environ["PROJECT_ENDPOINT"]
    model_deployment = os.environ["MODEL_DEPLOYMENT_NAME"]

    # Blob storage client for surfacing outputs via SAS download URLs
    blob_connection_string = os.environ["BLOB_STORAGE_CONNECTION_STRING"]
    storage_container_name = os.environ["BLOB_STORAGE_CONTAINER_NAME"]
    blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
    storage_container_client = blob_service_client.get_container_client(storage_container_name)

    # Foundry project + embedded OpenAI-compatible client (Responses API)
    project = AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential())
    openai = project.get_openai_client()

    # Upload input CSVs for the Code Interpreter container
    uploaded_files = [
        openai.files.create(purpose="assistants", file=open(csv_path, "rb"))
        for csv_path in args.csv
    ]

    # Create an agent version with Code Interpreter enabled and the input files preloaded
    agent = project.agents.create_version(
        agent_name="my-data-analyst-agent",
        definition=PromptAgentDefinition(
            model=model_deployment,
            instructions=(
                "You are a helpful data analyst agent. When the user's request involves "
                "tabular results, aggregations, or transformed datasets, save the output "
                "as a CSV file using the code interpreter and reference it in your reply. "
                "When the request calls for visualizations, also produce the relevant chart(s)."
            ),
            tools=[
                CodeInterpreterTool(
                    container=AutoCodeInterpreterToolParam(
                        file_ids=[f.id for f in uploaded_files]
                    )
                )
            ],
        ),
        description="Code interpreter agent for data analysis and visualization.",
    )

    conversation = openai.conversations.create()

    file_list = ", ".join(Path(p).name for p in args.csv)
    prompt_with_files = (
        f"{args.prompt}\n\nThe following files have been attached and are available "
        f"in the code interpreter workspace: {file_list}."
    )

    response = openai.responses.create(
        conversation=conversation.id,
        input=prompt_with_files,
        extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
    )

    print(f"Response status: {response.status.upper()}\n")

    def upload_and_link(local_path: Path, blob_name: str) -> str:
        with open(local_path, "rb") as data:
            storage_container_client.upload_blob(name=blob_name, data=data, overwrite=True)
        sas = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=storage_container_name,
            blob_name=blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=24),
        )
        return f"{storage_container_client.get_blob_client(blob_name).url}?{sas}"

    saved_file_ids: set[str] = set()

    def download_and_link(container_id: str, file_id: str, filename: str) -> None:
        if file_id in saved_file_ids:
            return
        saved_file_ids.add(file_id)
        safe_name = Path(filename).name or f"{file_id}.out"
        local_path = Path.cwd() / safe_name
        file_content = openai.containers.files.content.retrieve(
            file_id=file_id, container_id=container_id
        )
        with open(local_path, "wb") as f:
            f.write(file_content.read())
        print(f"Saved output file to: {local_path}")
        download_url = upload_and_link(local_path, safe_name)
        print(f"Download URL (24h): {download_url}\n")

    # Walk the response output: print assistant text, then download any
    # container_file_citation files (PNG charts, CSVs, etc.) referenced in annotations.
    input_file_ids = {f.id for f in uploaded_files}
    container_ids: set[str] = set()
    for item in response.output:
        item_type = getattr(item, "type", None)
        if item_type == "code_interpreter_call":
            container_id = getattr(item, "container_id", None)
            if container_id:
                container_ids.add(container_id)
            continue
        if item_type != "message":
            continue
        for content in item.content or []:
            if getattr(content, "type", None) != "output_text":
                continue
            print(f"ASSISTANT: {content.text}\n")
            for ann in content.annotations or []:
                if getattr(ann, "type", None) == "container_file_citation":
                    container_ids.add(ann.container_id)
                    download_and_link(ann.container_id, ann.file_id, ann.filename)

    # Fallback: enumerate every file in each container touched by this run and
    # download anything not already pulled in via citations (skipping input CSVs).
    for container_id in container_ids:
        for cf in openai.containers.files.list(container_id=container_id).data:
            if cf.id in saved_file_ids or cf.id in input_file_ids:
                continue
            filename = getattr(cf, "path", None) or getattr(cf, "filename", None) or cf.id
            download_and_link(container_id, cf.id, filename)

    # Clean up
    for f in uploaded_files:
        openai.files.delete(f.id)
    project.agents.delete_version(agent_name=agent.name, agent_version=agent.version)


if __name__ == "__main__":
    main()
