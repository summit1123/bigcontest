import vertexai
from vertexai.generative_models import GenerativeModel, Part

# TODO(developer): Update and un-comment below lines
# project_id = "PROJECT_ID"

vertexai.init(project=project_id, location="us-central1")

model = GenerativeModel(model_name="gemini-1.5-flash-001")

contents = [
    Part.from_uri(
        "gs://cloud-samples-data/generative-ai/video/pixel8.mp4",
        mime_type="video/mp4",
    ),
    "Provide a description of the video.",
]

# Prompt tokens count
response = model.count_tokens(contents)
print(f"Prompt Token Count: {response.total_tokens}")
print(f"Prompt Character Count: {response.total_billable_characters}")

# Send text to Gemini
response = model.generate_content(contents)
usage_metadata = response.usage_metadata

# Response tokens count
print(f"Prompt Token Count: {usage_metadata.prompt_token_count}")
print(f"Candidates Token Count: {usage_metadata.candidates_token_count}")
print(f"Total Token Count: {usage_metadata.total_token_count}")