import os
from pathlib import Path
from dotenv import load_dotenv

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import CodeInterpreterTool, FilePurpose, MessageRole

from azure.storage.blob import BlobServiceClient

def main():
    # Load environment variables from .env file
    load_dotenv()

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

    # Upload a data file for analysis
    file = project_client.agents.files.upload_and_poll(file_path="sample_dataset.csv", purpose=FilePurpose.AGENTS)

    # Initialize the code interpreter with the uploaded file
    code_interpreter = CodeInterpreterTool(file_ids=[file.id])

    # Create agent with code interpreter capabilities
    agent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="my-data-analyst-agent",
        instructions="You are helpful agent",
        tools=code_interpreter.definitions,
        tool_resources=code_interpreter.resources,
    )

    # Create conversation thread and initial message
    thread = project_client.agents.threads.create()
    message = project_client.agents.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="Could you please create bar chart with a breakdown of personnel types across all years in the dataset?",
    )

    # Process the message and execute code
    run = project_client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

    # Handle output files and annotations
    messages = project_client.agents.messages.list(thread_id=thread.id)

    for msg in messages:
        # Save every image file in the message and upload to blob storage
        for img in msg.image_contents:
            file_id = img.image_file.file_id
            file_name = f"{file_id}.png"
            project_client.agents.files.save(file_id=file_id, file_name=file_name)
            print(f"Saved image file to: {Path.cwd() / file_name}")
            with open(f"{Path.cwd() / file_name}", "rb") as data:
                storage_container_client.upload_blob(name=file_name, data=data, overwrite=True)
            print("Uploaded to blob storage.")

    # Clean up resources
    project_client.agents.files.delete(file.id)
    project_client.agents.delete_agent(agent.id)

if __name__ == "__main__":
    main()
