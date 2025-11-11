
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
import os, io, base64
from langchain_community.chat_models import ChatOllama
import asyncio
from src.logger_config import log

def encode_image_with_mime(image_path:str,resize_to):
    """Encodes all images in a folder to base64 with MIME type."""
    mime_strings = []
    # ✅ image_path is a directory
    if isinstance(image_path, str) and os.path.isdir(image_path):
        images_list = [
            os.path.join(image_path, f)
            for f in os.listdir(image_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    # Case 2: image_path is a single file
    elif isinstance(image_path, str) and os.path.isfile(image_path):
        images_list = [image_path]

    # Case 3: image_path is a list of file paths
    elif isinstance(image_path, list):
        images_list = image_path

    else:
        raise TypeError("image_path must be a valid file path, folder path, or list of paths")


    for img in images_list:
        #path = os.path.join(image_path,img)
        with Image.open(img) as image:
             # Resize directly
            image.thumbnail(resize_to, Image.Resampling.BILINEAR)

            # Save to memory buffer
            buffered = io.BytesIO()  #Creates a brand new empty "file in memory" where the resized image will be written.
            format = image.format.lower()
            if format in ["jpg", "jpeg"]:
                 format = "JPEG"
            elif format == "png":
                 format = "PNG"
            else:
                 format = image.format or "JPEG" 
            image.save(buffered, format=format.upper())

            # Convert to base64
            img_bytes = buffered.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            # Build final string
            mime_str = f"data:image/{format};base64,{img_b64}"
            mime_strings.append(mime_str)
    return mime_strings


async def process_single_image(llm, mime_str):
    """Process one image with the model."""
    sys_msg = SystemMessage(content="You are an image processing expert. Explain the image.")
    hum_msg = HumanMessage(content=[
        {"type": "text", "text": "Please describe this image:"},
        {"type": "image_url", "image_url": {"url": mime_str}}
    ])
    response = await llm.ainvoke([sys_msg, hum_msg])
    return response.content


async def extract_Image_summaries(image_path):
    """ Function to extract image sumamries"""
    try:
        llm = ChatOllama(model="qwen2.5vl:3b")
        mime_strings = encode_image_with_mime(image_path,resize_to=[224,224])

        # Run all requests in parallel
        tasks = [process_single_image(llm, mime_str) for mime_str in mime_strings]
        results = await asyncio.gather(*tasks)
        return results
    except Exception:
        log.exception("Extracting summary from Image failed")
        return []