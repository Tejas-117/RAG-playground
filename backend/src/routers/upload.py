from fastapi import APIRouter, File, UploadFile

router = APIRouter()

@router.post("/upload/")
async def upload_files(files: list[UploadFile] = File(...)) -> dict[str, list[str]]:
    """Print and return the names of files received from the ingestion form.

    Args:
        files: Multipart file fields submitted under the name ``files``.

    Returns:
        A dictionary containing every uploaded filename in submission order.
    """
    filenames = [file.filename or "unnamed" for file in files]

    for filename in filenames:
        print(filename, flush=True)

    for file in files:
        await file.close()

    return {"filenames": filenames}
