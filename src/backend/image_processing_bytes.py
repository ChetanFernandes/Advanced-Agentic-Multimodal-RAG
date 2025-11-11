from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
import io, base64
from langchain_community.chat_models import ChatOllama
import asyncio
from src.logger_config import log


def encode_image_with_mime(image_bytes,content_type,resize_to):
    """Encodes all images in a folder to base64 with MIME type."""
    mime_strings = []
    with Image.open(io.BytesIO(image_bytes)) as image:
        '''
        Here:
        The original bytes are parsed by Pillow
        The image now becomes a Python image object (PIL.Image.Image)
        This object contains raw pixel data in memory — a structured form, not encoded JPEG/PNG anymore. 
        ''' # io.BytesIO(image_bytes) is used when you have image data in memory'''
        # Resize directly
        image.thumbnail(resize_to, Image.Resampling.BILINEAR)

        # The resized image is written into a NEW buffer
        buffered = io.BytesIO()  #Creates a new in-memory “empty file” buffer where you’ll save the resized image (instead of writing to disk).
        
        format = content_type.rsplit("/",1)[1]
        if format in ["jpg", "jpeg"]:
            format = "JPEG"
        elif format == "png":
            format = "PNG"
        else:
            format = format or "JPEG" 

        image.save(buffered, format=format.upper())
        # Encodes the (resized) PIL image into the chosen file format and writes the bytes into the buffered in-memory file.

        # Convert to base64
        img_bytes = buffered.getvalue()

        # Reads all bytes currently stored in buffered and returns them as a bytes object.
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


async def extract_Image_summaries(image_bytes,content_type):
    """ Function to extract image sumamries"""
    try:
        llm = ChatOllama(model="qwen2.5vl:3b")
        mime_strings = encode_image_with_mime(image_bytes,content_type,resize_to=[224,224])

        # Run all requests in parallel
        tasks = [process_single_image(llm, mime_str) for mime_str in mime_strings]
        results = await asyncio.gather(*tasks)
        if not results:
            log.info("No summary extracted")
            return []
        
        return results
    
    except Exception:
        log.exception("Image summarization failed")
        return []