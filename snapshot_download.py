from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Salesforce/blip2-flan-t5-xl",
    local_dir="models/Salesforce/blip2-flan-t5-xl"
)