import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageSequence, ImageOps
from pathlib import Path
import numpy as np
import json
import trimesh as Trimesh
from tqdm import tqdm
import time
import shutil
import uuid
import math

import folder_paths
import node_helpers
import gc
import copy

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.utils

from .meshflow.pipelines import MeshFlowPipeline
from .meshflow.utils.dtype import AUTOCAST_DTYPE_CHOICES
from .meshflow.utils.mesh import (
    DEFAULT_NUM_VERTS,
    GEOMETRY_EXTS,
    Mesh,
    resolve_num_verts_for_mesh,
)

script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False

any = AnyType("*")

def get_filename_list(folder_path):
    # Get everything in the directory
    all_entries = os.listdir(folder_path)

    # Filter out directories to get only files
    filenames = [f for f in all_entries if os.path.isfile(os.path.join(folder_path, f))]

    return filenames

def tensor2pil(image: torch.Tensor) -> Image.Image:
    """
    Accepts either:
      - (H,W,C)
      - (1,H,W,C)
    Returns a PIL RGB/RGBA image depending on channels.
    """
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.ndim == 4:
            # Expect (B,H,W,C); allow only B==1 here
            if t.shape[0] != 1:
                raise ValueError(f"tensor2pil expects batch of 1, got batch={t.shape[0]}")
            t = t[0]
        elif t.ndim != 3:
            raise ValueError(f"tensor2pil expects (H,W,C) or (1,H,W,C), got shape={tuple(t.shape)}")

        arr = (t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    raise TypeError(f"tensor2pil expected torch.Tensor, got {type(image)}") 

class MeshflowLoadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {               
                "num_vertices": ("INT",{"default":4096,"max":65536,"min":64}),
            },
            "optional": {
                "dinov3_model": (get_filename_list(os.path.join(folder_paths.models_dir,'facebook')),),                
            }
        }

    RETURN_TYPES = ("MESHFLOWPIPELINE", )
    RETURN_NAMES = ("pipeline", )
    FUNCTION = "process"
    CATEGORY = "MeshflowWrapper"
    OUTPUT_NODE = True

    def process(self, num_vertices, dinov3_model = None):    
        model_dir_path = os.path.join(folder_paths.models_dir, 'facebook', 'meshflow')        
        
        if not os.path.exists(model_dir_path):
            print(f"Downloading model to: {model_dir_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id='facebook/meshflow',
                local_dir=model_dir_path,
                local_dir_use_symlinks=False,
            )

        model_path = os.path.join(model_dir_path,'meshflow')

        pipeline = MeshFlowPipeline.from_pretrained(
            model_path,
            device='cuda',
            dtype='bf16',
            num_verts=num_vertices,
            dinov3_model=dinov3_model
        )            
        
        return (pipeline,)

class MeshflowLoadMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {               
                "mesh_path": ("STRING",),
                "normalize": ("BOOLEAN",{"default":False}),
                "preprocess": ("BOOLEAN",{"default":False}),
            },
        }

    RETURN_TYPES = ("MESHFLOWMESH", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "MeshflowWrapper"
    OUTPUT_NODE = True

    def process(self, mesh_path, normalize, preprocess):  
        if not os.path.exists(mesh_path):
            mesh_path = os.path.join(comfy_path, 'input', mesh_path)

        mesh = Mesh.load_mesh(filename = mesh_path, normalize = normalize, preprocess = preprocess)        
        return (mesh,)
        
class MeshflowMeshProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {               
                "pipeline": ("MESHFLOWPIPELINE",),
                "mesh": ("MESHFLOWMESH",),
                "steps": ("INT",{"default":28,"min":10,"max":200,"step":1}),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0x7fffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "guidance_scale": ("FLOAT",{"default":2.5,"min":0.1,"max":9.9,"step":0.1}),
            }
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "MeshflowWrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, mesh, steps, seed, image = None, guidance_scale = 2.5):        
        if image is not None:
            pil_img = tensor2pil(image).convert("RGB")
        else:
            pil_img = None
        
        file_num_verts = int(mesh.verts.shape[0])
        if file_num_verts < int(pipeline.num_latents):
            proj_num_verts = file_num_verts
        else:
            proj_num_verts = pipeline.num_verts    
        
        denoiser = pipeline.models["denoiser"]
        if denoiser.use_proj_cond_on_temb:
            proj = pipeline.get_proj_cond_on_temb(
                num_verts=proj_num_verts,
                guidance_scale=guidance_scale,
            )
            print(
                f"  {mesh_path.name}: proj_num_verts={proj_num_verts} "
                f"proj_cond_on_temb={proj} (num_verts / num_latents={pipeline.num_latents})"
            )
            
        # RoPE must match denoiser input_size; proj_cond may use per-mesh vertex count.
        surface_pc = pipeline.sample_surface_points(mesh, num_verts=pipeline.num_verts)
        joint_kwargs = pipeline.get_rope_cond(surface_pc)
        visual_cond = pipeline.get_visual_cond(
            pil_img,
            guidance_scale=guidance_scale,
            preprocess_image=False,
        )
        
        latents = pipeline.sample_latent(
            visual_cond,
            joint_kwargs,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_verts=proj_num_verts,
        )
        
        out_mesh = pipeline.decode_latent(latents)
        out_trimesh = out_mesh.to_trimesh()
        
        return (out_trimesh,) 

class MeshflowExportMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "Meshflow"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            }
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("glb_path","relative_path",)
    FUNCTION = "process"
    CATEGORY = "MeshflowWrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, filename_prefix, file_format):        
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())                      
        output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
        output_glb_path.parent.mkdir(exist_ok=True)

        if file_format=='obj':
            materialName = f"{filename}_{counter:05}_.mtl"
            if hasattr(trimesh, 'visual') and hasattr(trimesh.visual, 'material') and trimesh.visual.material is not None:
                trimesh.visual.material.name = f"{filename}_{counter:05}"

            trimesh.export(output_glb_path, file_type=file_format, mtl_name=materialName)
        else:
            trimesh.export(output_glb_path, file_type=file_format)
            
        relative_path = Path(subfolder) / f'{filename}_{counter:05}_.{file_format}'
        
        return (str(output_glb_path), str(relative_path), )           
        
NODE_CLASS_MAPPINGS = {
    "MeshflowLoadModel": MeshflowLoadModel,
    "MeshflowLoadMesh": MeshflowLoadMesh,
    "MeshflowMeshProcessor": MeshflowMeshProcessor,
    "MeshflowExportMesh": MeshflowExportMesh,
    }
    

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshflowLoadModel": "Meshflow - LoadModel",
    "MeshflowLoadMesh": "Meshflow - LoadMesh",
    "MeshflowMeshProcessor": "Meshflow - MeshProcessor",
    "MeshflowExportMesh": "Meshflow - ExportMesh",
    }
