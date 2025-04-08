import asyncio

import bentoml
import base64
from pydantic import BaseModel
from typing import List
import aiohttp
import uuid
import subprocess

class FaceSwapRequest(BaseModel):
    source_image: str
    target_image: str


class FaceSwapModel:

    async def swap_face(self, input: FaceSwapRequest) -> str:
        """Perform face swapping between source and target image."""

        try:
            unique_id = str(uuid.uuid4())

            src_path = await self.decode_or_download_image(input.source_image, unique_id, "source")
            tgt_path = await self.decode_or_download_image(input.target_image, unique_id, "target")
            output_path = f"/tmp/output_{unique_id}.png"

            command = f"""
                    python facefusion.py headless-run --source-path {src_path} --target-path {tgt_path} --output-path {output_path} --execution-providers=cuda
                    """
            print(f"Running command: {command}")  # Debugging output

            process = await asyncio.create_subprocess_exec(
                "/bin/bash", "-c", command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            await process.communicate()

            with open(output_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        except Exception as e:
            return None

    async def decode_or_download_image(self, data: str, unique_id: str, image_type: str) -> str:
        """If data is a URL, download it. If it's base64, decode and save it."""
        if data.startswith("http"):
            return await self.download_image(data, unique_id, image_type)
        else:
            file_path = f"/tmp/{image_type}_{unique_id}.png"
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(data))
            return file_path  # Return file path

    async def download_image(self, url: str, unique_id: str, image_type: str) -> str:
        """Download an image from a URL asynchronously and save it to a unique temporary file."""
        file_path = f"/tmp/{image_type}_{unique_id}.png"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(file_path, "wb") as f:
                        f.write(await response.read())
        return file_path

bentoml.picklable_model.save_model("face_swap_model", FaceSwapModel())

@bentoml.service(
    traffic={
        "batch": True,
        "timeout": 300,
        "concurrency": 5,
        "max_batch_size": 5,
        "external_queue":False,
        "batch_wait": 0.1
    }
)
class FaceSwapBatchService:
    model = bentoml.picklable_model.load_model("face_swap_model")

    @bentoml.api(batchable=True)
    async def batch_face_swap(self, inputs: List[FaceSwapRequest]) -> List[dict]:
        print(f"Processing batch of size: {len(inputs)}")

        async def process_in_thread(input: FaceSwapRequest):
            return {
                "image": await asyncio.to_thread(self.model.swap_face, input)
            }

        # Process all in parallel using threads
        results = await asyncio.gather(*[process_in_thread(i) for i in inputs])
        return results


# Define the public API that handles incoming requests
@bentoml.service
class AIToolsAPI:
    face_swap_batch = bentoml.depends(FaceSwapBatchService)

    @bentoml.api
    async def face_swap(
            self,
            source_image: str = "",
            target_image: str = ""
    ) -> dict:

        result = await self.face_swap_batch.batch_face_swap([
            FaceSwapRequest(
                source_image=source_image,
                target_image=target_image
            )
        ])  # Process the input as part of a batch
        return result[0]

