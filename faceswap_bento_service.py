import asyncio
import base64
import uuid
import os
from typing import List
import rembg
import aiohttp
import bentoml
from pydantic import BaseModel


class FaceSwapRequest(BaseModel):
    source_image: str
    target_image: str

gpu_semaphore = asyncio.Semaphore(4)
class RemBGModel:
    async def rembg(self, input: source_image) -> dict:
        try:
            im = Image.open(BytesIO(base64.b64decode(input)))

            image = rembg.remove(
                im,
                session=rembg.new_session("isnet-general-use"),
                only_mask=False,
                alpha_matting=False,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10,
            )

            with io.BytesIO() as output_bytes:
                image.save(output_bytes, format="PNG")
                img_str = base64.b64encode(output_bytes.getvalue()).decode("utf-8")
            return {"image": img_str}
        except Exception as e:
            print(f"[EXCEPTION] Rembg failed: {e}")
            return None
class FaceSwapModel:
    async def swap_face(self, input: FaceSwapRequest) -> str:
       try:
            unique_id = str(uuid.uuid4())
            src_path = await self.decode_or_download_image(input.source_image, unique_id, "source")
            tgt_path = await self.decode_or_download_image(input.target_image, unique_id, "target")
            output_path = f"/tmp/output_{unique_id}.png"
            async with gpu_semaphore:
                process = await asyncio.create_subprocess_exec(
                    "python3", "facefusion.py", "headless-run",
                    "-s", src_path, "-t", tgt_path, "-o", output_path,
                    "--face-selector-order", "top-bottom",
                    "--processors", "face_swapper", "face_enhancer",
                    "--execution-providers", "cuda",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()

                stdout_text = stdout.decode().strip()
                stderr_text = stderr.decode().strip()

                print(f"[STDOUT]\n{stdout_text}")
                print(f"[STDERR]\n{stderr_text}")

                if process.returncode != 0:
                    print(f"[ERROR] Process exited with code {process.returncode}")
                    return None

                if not os.path.exists(output_path):
                    print(f"[ERROR] Output file not found at {output_path}")
                    return None

                with open(output_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
       except Exception as e:
            print(f"[EXCEPTION] Face swap failed: {e}")
            return None

    async def decode_or_download_image(self, data: str, unique_id: str, image_type: str) -> str:
        if data.startswith("http"):
            return await self.download_image(data, unique_id, image_type)
        else:
            file_path = f"/tmp/{image_type}_{unique_id}.png"
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(data))
            return file_path

    async def download_image(self, url: str, unique_id: str, image_type: str) -> str:
        file_path = f"/tmp/{image_type}_{unique_id}.png"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(file_path, "wb") as f:
                        f.write(await response.read())
                else:
                    raise ValueError(f"Failed to download image from URL: {url}")
        return file_path


@bentoml.service(
    traffic={
        "batch": True,
        "timeout": 300,
        "concurrency": 5,
        "max_batch_size": 5,
        "external_queue": False,
        "batch_wait": 0.1
    }
)
class FaceSwapBatchService:
    def __init__(self):
        self.model = FaceSwapModel()

    @bentoml.api()
    async def face_swap(self, input: FaceSwapRequest) -> dict:
        print("[INFO] Processing single face swap request")
        img_base64 = await self.model.swap_face(input)
        return {"image": img_base64}

    @bentoml.api(batchable=True)
    async def batch_face_swap(self, inputs: List[FaceSwapRequest]) -> List[dict]:
        print(f"[INFO] Processing batch of size: {len(inputs)}")

        async def process_one(input: FaceSwapRequest):
            img_base64 = await self.model.swap_face(input)
            return {"image": img_base64}

        return await asyncio.gather(*[process_one(i) for i in inputs], return_exceptions=True)

@bentoml.service(
    traffic={
        "batch": True,
        "timeout": 300,
        "concurrency": 5,
        "max_batch_size": 5,
        "external_queue": False,
        "batch_wait": 0.1
    }
)
class RemoveBgBatchService:
    def __init__(self):
        self.model = RemBGModel()

    @bentoml.api()
    async def rembg(self, input: str) -> dict:
        print("[INFO] Processing single Rembg request")
        response = await self.model.rembg(input)
        return response

@bentoml.service
class AIToolsAPI:
    face_swap_batch = bentoml.depends(FaceSwapBatchService)
    rembg_batch = bentoml.depends(RemoveBgBatchService)

    @bentoml.api
    async def faceswap(self, source_image: str = "", target_image: str = "") -> dict:
        result = await self.face_swap_batch.face_swap(
            FaceSwapRequest(source_image=source_image, target_image=target_image)
        )
        return result

    @bentoml.api
    async def rembg(self, source_image: str = "") -> dict:
        result = await self.rembg_batch.rembg(source_image)
        return result
