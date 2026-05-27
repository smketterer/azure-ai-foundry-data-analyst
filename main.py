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

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"


def _label(color: str, text: str) -> str:
    return f"{BOLD}{color}{text}{RESET}"


AGENT_INSTRUCTIONS = (
    "You are a helpful data analyst agent. When the user's request involves "
    "tabular results, aggregations, or transformed datasets, save the output "
    "as a CSV file using the code interpreter and reference it in your reply. "
    "When the request calls for visualizations, also produce the relevant chart(s)."
)

PLAN_PREFIX = (
    "PLANNING PHASE. Do NOT execute any code yet. "
    "Write a concise, numbered plan (3-6 steps) describing exactly how you will address "
    "the user's request below, including which columns/files you'll use, what analyses "
    "you'll run, and which artifacts (charts/CSVs) you'll produce.\n\n"
    "USER REQUEST:\n"
)

EXECUTE_PREFIX = (
    "EXECUTION PHASE. Follow the plan you just produced. Run the code interpreter as "
    "needed, save any tabular outputs as CSVs and any visualizations as charts, and "
    "summarize the results for the user."
)

CRITIQUE_PREFIX = (
    "CRITIQUE PHASE. Critically review your most recent execution. "
    "Identify any correctness issues, questionable assumptions, missing edge cases, "
    "data-quality concerns, or improvements to the artifacts. If everything is sound, "
    "explicitly say 'No revisions needed.' Otherwise list the specific issues."
)

REVISE_PREFIX = (
    "REVISION PHASE. Address every issue you raised in the critique. Re-run the code "
    "interpreter as needed and produce updated artifacts. If no revisions were needed, "
    "say so and stop."
)

FINAL_PREFIX = (
    "FINAL PHASE. Produce the user-facing final response. Write a concise summary of "
    "the conclusions, then explicitly cite every final artifact the user should keep "
    "(charts, CSVs, etc.) so each is included as a downloadable file citation. Do not "
    "include intermediate or superseded artifacts."
)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run an Azure AI Foundry data-analyst agent over one or more CSV files."
    )
    parser.add_argument(
        "--mode",
        choices=["oneshot", "repl"],
        default="oneshot",
        help="oneshot: single prompt, single response. repl: multi-turn with plan/critique/revise loop.",
    )
    parser.add_argument(
        "--prompt",
        help="The user prompt to send to the agent. Required for oneshot; optional initial prompt for repl.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        nargs="+",
        help="One or more CSV file paths to upload for analysis.",
    )
    args = parser.parse_args()

    if args.mode == "oneshot" and not args.prompt:
        parser.error("--prompt is required when --mode=oneshot")

    project_endpoint = os.environ["PROJECT_ENDPOINT"]
    model_deployment = os.environ["MODEL_DEPLOYMENT_NAME"]

    blob_connection_string = os.environ["BLOB_STORAGE_CONNECTION_STRING"]
    storage_container_name = os.environ["BLOB_STORAGE_CONTAINER_NAME"]
    blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
    storage_container_client = blob_service_client.get_container_client(storage_container_name)

    project = AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential())
    openai = project.get_openai_client()

    uploaded_files = [
        openai.files.create(purpose="assistants", file=open(csv_path, "rb"))
        for csv_path in args.csv
    ]

    agent = project.agents.create_version(
        agent_name="my-data-analyst-agent",
        definition=PromptAgentDefinition(
            model=model_deployment,
            instructions=AGENT_INSTRUCTIONS,
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
    file_note = (
        f"\n\nThe following files have been attached and are available "
        f"in the code interpreter workspace: {file_list}."
    )

    outputs_dir = Path.cwd() / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    saved_file_ids: set[str] = set()

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

    def download_and_link(container_id: str, file_id: str, filename: str) -> None:
        if file_id in saved_file_ids:
            return
        saved_file_ids.add(file_id)
        safe_name = Path(filename).name or f"{file_id}.out"
        local_path = outputs_dir / safe_name
        file_content = openai.containers.files.content.retrieve(
            file_id=file_id, container_id=container_id
        )
        with open(local_path, "wb") as f:
            f.write(file_content.read())
        print(f"{_label(MAGENTA, 'Saved:')} {local_path}")
        download_url = upload_and_link(local_path, safe_name)
        print(f"{_label(MAGENTA, 'Download (24h):')} {download_url}\n")

    def run_phase(
        label: str, color: str, input_text: str, download_files: bool
    ) -> tuple[str, list[tuple[str, str, str]]]:
        print(f"{_label(color, f'[{label}]')}")
        response = openai.responses.create(
            conversation=conversation.id,
            input=input_text,
            extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
        )
        text_chunks: list[str] = []
        citations: list[tuple[str, str, str]] = []
        for item in response.output:
            if getattr(item, "type", None) != "message":
                continue
            for content in item.content or []:
                if getattr(content, "type", None) != "output_text":
                    continue
                print(content.text + "\n")
                text_chunks.append(content.text)
                for ann in content.annotations or []:
                    if getattr(ann, "type", None) == "container_file_citation":
                        citations.append((ann.container_id, ann.file_id, ann.filename))
                        if download_files:
                            download_and_link(ann.container_id, ann.file_id, ann.filename)
        return "\n".join(text_chunks), citations

    def run_turn(user_prompt: str, with_file_note: bool) -> None:
        full_prompt = user_prompt + (file_note if with_file_note else "")
        print(f"{_label(CYAN, '[USER]')}\n{full_prompt}\n")

        if args.mode == "oneshot":
            run_phase("ASSISTANT", GREEN, full_prompt, download_files=True)
            return

        # REPL: plan -> execute -> critique -> (maybe) revise -> final.
        # Artifacts are saved only after the final phase, using citations from
        # the latest artifact-producing phase (revise if it ran, else execute).
        run_phase("PLAN", BLUE, PLAN_PREFIX + full_prompt, download_files=False)
        _, execute_citations = run_phase("EXECUTE", GREEN, EXECUTE_PREFIX, download_files=False)
        critique, _ = run_phase("CRITIQUE", YELLOW, CRITIQUE_PREFIX, download_files=False)
        final_citations = execute_citations
        if "no revisions needed" not in critique.lower():
            _, revise_citations = run_phase("REVISE", MAGENTA, REVISE_PREFIX, download_files=False)
            if revise_citations:
                final_citations = revise_citations
        run_phase("FINAL", GREEN, FINAL_PREFIX, download_files=False)
        for container_id, file_id, filename in final_citations:
            download_and_link(container_id, file_id, filename)

    try:
        if args.mode == "oneshot":
            run_turn(args.prompt, with_file_note=True)
            return

        # REPL mode
        print(
            f"{_label(BLUE, 'REPL mode.')} Type a prompt and press Enter. "
            "Commands: 'exit' or 'quit' to leave.\n"
        )
        first_turn = True
        if args.prompt:
            run_turn(args.prompt, with_file_note=True)
            first_turn = False

        while True:
            try:
                user_input = input(f"{_label(CYAN, '>')} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break
            run_turn(user_input, with_file_note=first_turn)
            first_turn = False
    finally:
        for f in uploaded_files:
            try:
                openai.files.delete(f.id)
            except Exception as exc:
                print(f"{_label(YELLOW, 'Cleanup warning:')} failed to delete file {f.id}: {exc}")
        try:
            project.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
        except Exception as exc:
            print(f"{_label(YELLOW, 'Cleanup warning:')} failed to delete agent version: {exc}")


if __name__ == "__main__":
    main()
