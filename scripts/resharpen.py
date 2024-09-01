from modules.processing import StableDiffusionProcessingTxt2Img as txt2img
from modules.processing import StableDiffusionProcessingImg2Img as img2img
from modules.sd_samplers_kdiffusion import KDiffusionSampler as KSampler
from modules.script_callbacks import on_script_unloaded
from modules.ui_components import InputAccordion
from modules import scripts

from lib_resharpen.scaling import apply_scaling
from lib_resharpen.param import ReSharpenParams
from lib_resharpen.xyz import xyz_support

from functools import wraps
from typing import Callable
import gradio as gr
import torch


original_callback: Callable = KSampler.callback_state


@torch.inference_mode()
@wraps(original_callback)
def hijack_callback(self, d: dict):
    if getattr(self.p, "_ad_inner", False):
        return original_callback(self, d)

    params: ReSharpenParams = getattr(self, "resharpen_params", None)

    if not params:
        return original_callback(self, d)

    if params.cache is not None:
        delta: torch.Tensor = d["x"].detach().clone() - params.cache
        d["x"] += delta * apply_scaling(
            params.scaling, params.strength, d["i"], params.total_step
        )

    params.cache = d["x"].detach().clone()
    return original_callback(self, d)


KSampler.callback_state = hijack_callback


class ReSharpen(scripts.Script):

    def __init__(self):
        self.XYZ_CACHE = {}
        xyz_support(self.XYZ_CACHE)

    def title(self):
        return "ReSharpen"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(value=False, label=self.title()) as enable:

            gr.Markdown(
                '<h3 style="float: left;">Softer</h3> <h3 style="float: right;">Sharper</h3>'
            )

            with gr.Group(elem_classes="resharpen"):
                decay = gr.Slider(
                    label="Sharpness", minimum=-1.0, maximum=1.0, step=0.1, value=0.0
                )
                scaling = gr.Radio(
                    ["Flat", "Cos", "Sin", "1 - Cos", "1 - Sin"],
                    label="Scaling Settings",
                    value="Flat",
                )

            with gr.Group(elem_classes=["resharpen", "hr"], visible=(not is_img2img)):
                hr_decay = gr.Slider(
                    label="Hires. Fix Sharpness",
                    minimum=-1.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.0,
                )
                hr_scaling = gr.Radio(
                    ["Flat", "Cos", "Sin", "1 - Cos", "1 - Sin"],
                    label="Scaling Settings",
                    value="Flat",
                )

        self.paste_field_names = []
        self.infotext_fields = [
            (enable, "Resharpen Enabled"),
            (decay, "Resharpen Sharpness"),
            (scaling, "Resharpen Scaling"),
            (hr_decay, "Resharpen Sharpness Hires"),
            (hr_scaling, "Resharpen Scaling Hires"),
        ]

        for comp, name in self.infotext_fields:
            comp.do_not_save_to_config = True
            self.paste_field_names.append(name)

        return [enable, decay, scaling, hr_decay, hr_scaling]

    def process(
        self,
        p,
        enable: bool,
        decay: float,
        scaling: str,
        hr_decay: float,
        hr_scaling: str,
        *args,
        **kwargs,
    ):

        setattr(KSampler, "resharpen_params", None)
        
        if not enable:
            self.XYZ_CACHE.clear()
            return p

        if p.sampler_name.strip() == "Euler a":
            print("\n[ReSharpen] has little effect with Ancestral samplers!\n")

        decay = float(self.XYZ_CACHE.pop("decay", decay))
        scaling = str(self.XYZ_CACHE.pop("scaling", scaling))

        p.extra_generation_params.update(
            {
                "Resharpen Enabled": enable,
                "Resharpen Sharpness": decay,
                "Resharpen Scaling": scaling,
            }
        )

        params = ReSharpenParams(
            enable,
            scaling,
            decay / -10.0,
            0,
            None,
        )

        setattr(KSampler, "resharpen_params", params)

        if isinstance(p, img2img):
            self.XYZ_CACHE.clear()
            return p

        assert isinstance(p, txt2img)

        if not getattr(p, "enable_hr", False):
            self.XYZ_CACHE.clear()
            return p

        hr_decay = float(self.XYZ_CACHE.get("hr_decay", hr_decay))
        hr_scaling = str(self.XYZ_CACHE.get("hr_scaling", hr_scaling))
        p.extra_generation_params.update(
            {
                "Resharpen Sharpness Hires": hr_decay,
                "Resharpen Scaling Hires": hr_scaling,
            }
        )

        return p

    def process_before_every_sampling(self, p, enable, *args, **kwargs):
        if enable and hasattr(KSampler, "resharpen_params"):
            KSampler.resharpen_params.total_step = (
                getattr(p, "firstpass_steps", None) or p.steps
            )

    def before_hr(
        self,
        p,
        enable: bool,
        decay: float,
        scaling: str,
        hr_decay: float,
        hr_scaling: str,
        *args,
        **kwargs,
    ):
        if not enable:
            return p

        hr_decay = float(self.XYZ_CACHE.pop("hr_decay", hr_decay))
        hr_scaling = str(self.XYZ_CACHE.pop("hr_scaling", hr_scaling))

        params = ReSharpenParams(
            enable,
            hr_scaling,
            hr_decay / -10.0,
            getattr(p, "hr_second_pass_steps", None) or p.steps,
            None,
        )

        setattr(KSampler, "resharpen_params", params)

        self.XYZ_CACHE.clear()
        return p


def restore_callback():
    KSampler.callback_state = original_callback


on_script_unloaded(restore_callback)
