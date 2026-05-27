import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import CodeInterpreterTool, FilePurpose, MessageRole

from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas

def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run an Azure AI Foundry data-analyst agent over one or more CSV files.")
    parser.add_argument("--prompt", required=True, help="The user prompt to send to the agent.")
    parser.add_argument(
        "--csv",
        required=True,
        nargs="+",
        help="One or more CSV file paths to upload for analysis.",
    )
    args = parser.parse_args()

    # Create an Azure AI Client from an endpoint, copied from your Azure AI Foundry project.
    # You need to login to Azure subscription via Azure CLI and set the environment variables
    project_endpoint = os.environ["PROJECT_ENDPOINT"]  # Ensure the PROJECT_ENDPOINT environment variable is set

    # Create a blob storage client using connection string
    blob_storage_connection_string = os.environ["BLOB_STORAGE_CONNECTION_STRING"]
    storage_container_name = os.environ["BLOB_STORAGE_CONTAINER_NAME"]
    blob_service_client = BlobServiceClient.from_connection_string(blob_storage_connection_string)
    storage_container_client = blob_service_client.get_container_client(storage_container_name)

    # Create an AIProjectClient instance
    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(),  # Use Azure Default Credential for authentication
    )

    # Upload data files for analysis
    uploaded_files = [
        project_client.agents.files.upload_and_poll(file_path=csv_path, purpose=FilePurpose.AGENTS)
        for csv_path in args.csv
    ]

    # Initialize the code interpreter with the uploaded files
    code_interpreter = CodeInterpreterTool(file_ids=[f.id for f in uploaded_files])

    # Create agent with code interpreter capabilities
    agent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="my-data-analyst-agent",
        instructions=(
            "You are a helpful data analyst agent. When the user's request involves "
            "tabular results, aggregations, or transformed datasets, save the output "
            "as a CSV file using the code interpreter and reference it in your reply. "
            "When the request calls for visualizations, also produce the relevant chart(s)."
        ),
        tools=code_interpreter.definitions,
        tool_resources=code_interpreter.resources,
    )

    # Create conversation thread and initial message
    thread = project_client.agents.threads.create()
    message = project_client.agents.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=args.prompt,
    )

    # Process the message and execute code
    run = project_client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
    print(f"Run finished with status: {run.status.value.upper()}\n")

    if run.status == "failed":
        # Log the error if the run fails
        print(f"Run failed: {run.last_error}")
    
    uploaded_input_ids = {f.id for f in uploaded_files}

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

    # Fetch and print all messages
    messages = project_client.agents.messages.list(thread_id=thread.id)
    for msg in messages:
        for content in msg.content:
            if content.type == "text":
                print(f"{msg.role.value.upper()}: {content["text"]["value"]}\n")
        # Save every image file in the message and upload to blob storage
        for img in msg.image_contents:
            file_id = img.image_file.file_id
            file_name = f"{file_id}.png"
            project_client.agents.files.save(file_id=file_id, file_name=file_name)
            local_path = Path.cwd() / file_name
            print(f"Saved image file to: {local_path}")
            download_url = upload_and_link(local_path, file_name)
            print(f"Download URL (24h): {download_url}\n")
        # Save every generated file (e.g. CSV) referenced by file_path annotations
        for annotation in msg.file_path_annotations:
            file_id = annotation.file_path.file_id
            if file_id in uploaded_input_ids:
                continue
            file_name = Path(annotation.text).name or f"{file_id}.out"
            project_client.agents.files.save(file_id=file_id, file_name=file_name)
            local_path = Path.cwd() / file_name
            print(f"Saved output file to: {local_path}")
            download_url = upload_and_link(local_path, file_name)
            print(f"Download URL (24h): {download_url}\n")

    # Clean up resources
    for f in uploaded_files:
        project_client.agents.files.delete(f.id)
    project_client.agents.delete_agent(agent.id)

if __name__ == "__main__":
    main()
