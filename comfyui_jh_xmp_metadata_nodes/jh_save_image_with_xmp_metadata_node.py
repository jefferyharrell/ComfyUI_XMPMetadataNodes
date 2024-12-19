"""
Module for saving images with XMP metadata.

This module provides classes and methods to handle image saving operations,
including embedding XMP metadata into supported image formats.

Classes:
    JHSupportedImageTypes: Enum representing supported image formats.
    JHSaveImageWithXMPMetadataNode: Handles saving images with metadata.
"""

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import folder_paths  # pyright: ignore[reportMissingImports]
import numpy as np
import PIL.Image
import torch
from PIL.Image import Image
from PIL.PngImagePlugin import PngInfo

from .jh_xmp_metadata import JHXMPMetadata


class JHSupportedImageTypes(StrEnum):
    """
    Enumeration representing supported image types.

    This enum defines a set of image formats that are supported for saving with
    metadata, including standard formats as well as specialized formats with additional
    metadata.

    Attributes:
        JPEG: Represents the JPEG format, a commonly used lossy compressed image format.
        PNG_WITH_WORKFLOW: Represents a PNG file with an embedded ComfyUI workflow.
        PNG: Represents the standard PNG format, a lossless image compression format.
        LOSSLESS_WEBP: Represents the WebP format in its lossless variant.
        WEBP: Represents the standard lossy WebP format.
    """

    JPEG = "JPEG"
    PNG_WITH_WORKFLOW = "PNG with embedded workflow"
    PNG = "PNG"
    LOSSLESS_WEBP = "Lossless WebP"
    WEBP = "WebP"


class JHSaveImageWithXMPMetadataNode:
    """
    A node for saving images with XMP metadata.

    This class is designed to handle the saving of images with optional XMP metadata.
    It supports multiple image formats and provides flexibility for embedding metadata
    like creator, title, description, and custom XML strings.

    Attributes:
        output_dir (str): The directory where images will be saved.
        type (str): The type of node, typically 'output'.
        prefix_append (str): A string to append to filename prefixes.
        compress_level (int): Compression level for saving images (applicable for some
            formats).
    """

    def __init__(self, output_dir: str | None = None) -> None:
        self.output_dir: str = (
            output_dir
            if output_dir is not None
            else folder_paths.get_output_directory()
        )
        self.type: str = "output"
        self.prefix_append: str = ""
        self.compress_level: int = 0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # pylint: disable=invalid-name
        """
        Define the input types and their configuration for the node.

        Returns:
            dict: A dictionary of required, optional, and hidden input types.
        """
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": (
                            "The prefix for the file to save. This may include "
                            "formatting information such as %date:yyyy-MM-dd% or "
                            "%Empty Latent Image.width% to include values from nodes."
                        ),
                    },
                ),
                "image_type": (
                    [x for x in JHSupportedImageTypes],
                    {
                        "default": JHSupportedImageTypes.PNG_WITH_WORKFLOW,
                    },
                ),
            },
            "optional": {
                "creator": (
                    "STRING",
                    {"tooltip": ("dc:creator"), "forceInput": True},
                ),
                "title": (
                    "STRING",
                    {"tooltip": ("dc:title"), "forceInput": True},
                ),
                "description": (
                    "STRING",
                    {"tooltip": ("dc:description"), "forceInput": True},
                ),
                "subject": (
                    "STRING",
                    {"tooltip": ("dc:subject"), "forceInput": True},
                ),
                "instructions": (
                    "STRING",
                    {"tooltip": ("photoshop:Instructions"), "forceInput": True},
                ),
                "comment": (
                    "STRING",
                    {"tooltip": ("exif:UserComment"), "forceInput": True},
                ),
                "alt_text": (
                    "STRING",
                    {
                        "tooltip": ("Iptc4xmpCore:AltTextAccessibility"),
                        "forceInput": True,
                    },
                ),
                "xml_string": (
                    "STRING",
                    {
                        "tooltip": (
                            "XMP metadata as an XML string. This will override all "
                            "other fields."
                        ),
                        "forceInput": True,
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save_images"
    CATEGORY = "XMP Metadata Nodes"
    OUTPUT_NODE = True

    def save_images(
        self,
        images: list,
        filename_prefix: str = "ComfyUI",
        image_type: JHSupportedImageTypes = JHSupportedImageTypes.PNG_WITH_WORKFLOW,
        creator: str | list | None = None,
        title: str | list | None = None,
        description: str | list | None = None,
        subject: str | list | None = None,
        instructions: str | list | None = None,
        comment: str | list | None = None,
        alt_text: str | list | None = None,
        xml_string: str | None = None,
        prompt: str | None = None,
        extra_pnginfo: dict | None = None,
    ) -> dict:
        """
        Saves a batch of images to disk with metadata and specified formats.

        This method processes a list of images, embeds metadata such as creator,
        title, description, and XMP workflow, and saves them in the desired file
        format and directory structure.

        Args:
            images (list): List of images to save, where each image is a numpy array.
            filename_prefix (str): Prefix for the saved file names. Default is
                "ComfyUI".
            image_type (JHSupportedImageTypes): Format in which to save the images
                (e.g., JPEG, PNG, WebP). Defaults to PNG with embedded workflow.
            creator (str | list | None): Creator metadata. Can be a string or a list
                matching the batch size.
            title (str | list | None): Title metadata. Can be a string or a list
                matching the batch size.
            description (str | list | None): Description metadata. Can be a string or a
                list matching the batch size.
            subject (str | list | None): Subject metadata. Can be a string or a list
                matching the batch size.
            instructions (str | list | None): Instructions metadata. Can be a string or
                a list matching the batch size.
            comment (str | list | None): Comment metadata. Can be a string or a list
                matching the batch size.
            alt_text (str | list | None): Alt text metadata. Can be a string or a list
                matching the batch size.
            xml_string (str | None): Optional pre-generated XML metadata string. If
                provided, this overrides the individual metadata fields.
            prompt (str | None): Prompt metadata to embed in the image (if applicable).
            extra_pnginfo (dict | None): Additional PNG metadata, such as workflow
                information.

        Returns:
            dict: A dictionary containing the saved image paths and metadata, including:
                - "result": The original list of images.
                - "ui": Metadata for each saved image, including filename, subfolder,
                  and type.

        Raises:
            ValueError: If no images are provided or if an unsupported image type is
                specified.
        """
        if images is None or len(images) == 0:
            raise ValueError("No images to save.")

        filename_prefix += self.prefix_append
        full_output_folder: str
        filename: str
        counter: int
        subfolder: str
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results: list = []

        filename_extension: str = self.extension_for_type(image_type)

        batch_number: int = 0
        image: torch.Tensor

        for batch_number, image in enumerate(images):
            i: np.ndarray = 255.0 * image.cpu().numpy()
            img: Image = PIL.Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            filename_with_batch_num: str = filename.replace(
                "%batch_num%", str(batch_number)
            )
            file: str = f"{filename_with_batch_num}_{counter:05}_.{filename_extension}"

            xmp = self.xmp(
                creator,
                title,
                description,
                subject,
                instructions,
                comment,
                alt_text,
                xml_string,
                batch_number,
            )

            self.save_image(
                img,
                image_type,
                Path(full_output_folder) / file,
                xmp,
                prompt,
                extra_pnginfo,
            )

            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"result": (images,), "ui": {"images": results}}

    def xmp(
        self,
        creator: str | list | None,
        title: str | list | None,
        description: str | list | None,
        subject: str | list | None,
        instructions: str | list | None,
        comment: str | list | None,
        alt_text: str | list | None,
        xml_string: str | None,
        batch_number: int,
    ) -> str:
        """
        Return XMP metadata as a string.

        If xml_string is provided, use it as the XMP metadata. Otherwise, create
        a JHXMPMetadata object and populate it with the provided metadata fields.
        If the fields are lists, use the value at the index given by batch_number.
        If the fields are single values, use the same value for all images.
        Return the XMP metadata as a string wrapped in a <?xpacket begin="..."?>
        tag.
        """

        if xml_string is not None:
            xmp: str = xml_string
        else:
            xmpmetadata = JHXMPMetadata()
            xmpmetadata.creator = (
                creator[batch_number] if isinstance(creator, list) else creator
            )
            xmpmetadata.title = (
                title[batch_number] if isinstance(title, list) else title
            )
            xmpmetadata.description = (
                description[batch_number]
                if isinstance(description, list)
                else description
            )
            xmpmetadata.subject = (
                subject[batch_number] if isinstance(subject, list) else subject
            )
            xmpmetadata.instructions = (
                instructions[batch_number]
                if isinstance(instructions, list)
                else instructions
            )
            xmpmetadata.comment = (
                comment[batch_number] if isinstance(comment, list) else comment
            )
            xmpmetadata.alt_text = (
                alt_text[batch_number] if isinstance(alt_text, list) else alt_text
            )
            xmp = xmpmetadata.to_wrapped_string()
        return xmp

    def extension_for_type(self, image_type: JHSupportedImageTypes) -> str:
        """
        Determines the file extension for a given image type.

        Args:
            image_type (JHSupportedImageTypes): The type of the image.

        Returns:
            str: The file extension corresponding to the given image type.

        Raises:
            ValueError: If the provided image type is not supported.
        """
        filename_extension: str
        match image_type:
            case JHSupportedImageTypes.JPEG:
                filename_extension: str = "jpg"
            case JHSupportedImageTypes.PNG_WITH_WORKFLOW:
                filename_extension: str = "png"
            case JHSupportedImageTypes.PNG:
                filename_extension: str = "png"
            case JHSupportedImageTypes.LOSSLESS_WEBP:
                filename_extension: str = "webp"
            case JHSupportedImageTypes.WEBP:
                filename_extension: str = "webp"
            case _:
                raise ValueError(f"Unsupported image type: {image_type}")
        return filename_extension

    def save_image(
        self,
        image: Image,
        image_type: JHSupportedImageTypes,
        to_path: Path,
        xmp: str,
        prompt: str | None = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ) -> None:
        """
        Saves an image to the specified path with embedded XMP metadata.

        This method handles different image formats and embeds XMP metadata
        and optional additional information such as prompts and workflows.

        Args:
            image (Image): The image to be saved.
            image_type (JHSupportedImageTypes): The format in which to save the image.
            to_path (Path): The file path where the image will be saved.
            xmp (str): The XMP metadata, as an XML string, to embed in the image.
            prompt (Optional[str]): Optional prompt metadata to include in PNG images.
            extra_pnginfo (Optional[dict]): Additional PNG metadata such as workflow
                information.

        Raises:
            ValueError: If the provided image type is not supported.
        """
        match image_type:
            case JHSupportedImageTypes.PNG_WITH_WORKFLOW:
                pnginfo: PngInfo = PngInfo()
                pnginfo.add_text("XML:com.adobe.xmp", xmp)
                if prompt is not None:
                    pnginfo.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    pnginfo.add_text("workflow", json.dumps(extra_pnginfo["workflow"]))
                image.save(
                    to_path,
                    pnginfo=pnginfo,
                    compress_level=self.compress_level,
                )

            case JHSupportedImageTypes.PNG:
                pnginfo: PngInfo = PngInfo()
                pnginfo.add_text("XML:com.adobe.xmp", xmp)
                image.save(
                    to_path,
                    pnginfo=pnginfo,
                    compress_level=self.compress_level,
                )

            case JHSupportedImageTypes.JPEG:
                image.save(
                    to_path,
                    xmp=xmp.encode("utf-8"),
                )

            case JHSupportedImageTypes.LOSSLESS_WEBP:
                image.save(
                    to_path,
                    xmp=xmp,
                    lossless=True,
                )

            case JHSupportedImageTypes.WEBP:
                image.save(to_path, xmp=xmp)

            case _:
                raise ValueError(f"Unsupported image type: {image_type}")
