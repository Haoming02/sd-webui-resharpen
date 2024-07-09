from modules.sd_samplers_kdiffusion import KDiffusionSampler
from scripts.res_scaling import apply_scaling
from scripts.res_xyz import xyz_support
from modules import script_callbacks
from modules.shared import opts

import modules.scripts as scripts
import gradio as gr

from modules.processing import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
)

original_callback = KDiffusionSampler.callback_state


def hijack_callback(self, d):
    if not getattr(self, "trajectory_enable", False):
        return original_callback(self, d)

    if getattr(self.p, "_ad_inner", False):
        return original_callback(self, d)

    if self.traj_cache is not None:
        delta = d["x"].detach().clone() - self.traj_cache
        d["x"] += delta * apply_scaling(
            self.traj_scaling, self.traj_decay, d["i"], self.traj_totalsteps
        )

    self.traj_cache = d["x"].detach().clone()
    return original_callback(self, d)


KDiffusionSampler.callback_state = hijack_callback


class ReSharpen(scripts.Script):
    def __init__(self):
        self.XYZ_CACHE = {}
        xyz_support(self.XYZ_CACHE)

    def title(self):
        return "ReSharpen"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("ReSharpen", open=False):
            enable = gr.Checkbox(label="Enable")

            gr.Markdown(
                '<h3 style="float: left;">Softer</h3> <h3 style="float: right;">Sharper</h3>'
            )

            decay = gr.Slider(
                label="Sharpness", minimum=-1.0, maximum=1.0, step=0.1, value=0.0
            )
            scaling = gr.Radio(
                ["Flat", "Cos", "Sin", "1 - Cos", "1 - Sin"],
                label="Scaling Settings",
                value="Flat",
            )

            if not is_img2img:
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
        ]

        if not is_img2img:
            self.infotext_fields.append((hr_decay, "Resharpen Sharpness Hires"))
            self.infotext_fields.append((hr_scaling, "Resharpen Scaling Hires"))

        for comp, name in self.infotext_fields:
            comp.do_not_save_to_config = True
            self.paste_field_names.append(name)

        if not is_img2img:
            return [enable, decay, scaling, hr_decay, hr_scaling]
        else:
            return [enable, decay, scaling, decay, scaling]

    def process(
        self,
        p,
        enable: bool,
        decay: float,
        scaling: str,
        hr_decay: float,
        hr_scaling: str,
    ):
        setattr(KDiffusionSampler, "trajectory_enable", enable)
        if not enable:
            self.XYZ_CACHE.clear()
            return p

        if "decay" in self.XYZ_CACHE:
            decay = float(self.XYZ_CACHE["decay"])
            del self.XYZ_CACHE["decay"]

        if "scaling" in self.XYZ_CACHE:
            scaling = str(self.XYZ_CACHE["scaling"])
            del self.XYZ_CACHE["scaling"]

        if enable:
            p.extra_generation_params["Resharpen Enabled"] = enable
            p.extra_generation_params["Resharpen Sharpness"] = decay
            p.extra_generation_params["Resharpen Scaling"] = scaling

            if p.sampler_name.strip() == "Euler a":
                print("\n[Resharpen] has little effect with Ancestral samplers!\n")

            KDiffusionSampler.traj_decay = decay / -10.0
            KDiffusionSampler.traj_cache = None
            KDiffusionSampler.traj_scaling = scaling

            if isinstance(p, StableDiffusionProcessingImg2Img):
                self.XYZ_CACHE.clear()
                dnstr: float = getattr(p, "denoising_strength", 1.0)
                if dnstr < 1.0 and not getattr(opts, "img2img_fix_steps", False):
                    KDiffusionSampler.traj_totalsteps = int(p.steps * dnstr + 1.0)

            else:
                assert isinstance(p, StableDiffusionProcessingTxt2Img)
                KDiffusionSampler.traj_totalsteps = p.steps

                if getattr(p, "enable_hr", False):
                    if "hr_decay" in self.XYZ_CACHE:
                        hr_decay = float(self.XYZ_CACHE["hr_decay"])
                    if "hr_scaling" in self.XYZ_CACHE:
                        hr_scaling = str(self.XYZ_CACHE["hr_scaling"])

                    p.extra_generation_params["Resharpen Sharpness Hires"] = hr_decay
                    p.extra_generation_params["Resharpen Scaling Hires"] = hr_scaling

                else:
                    self.XYZ_CACHE.clear()

        assert len(self.XYZ_CACHE) <= 2
        return p

    def before_hr(
        self,
        p,
        enable: bool,
        decay: float,
        scaling: str,
        hr_decay: float,
        hr_scaling: str,
    ):
        if not enable:
            return p

        if "hr_decay" in self.XYZ_CACHE:
            hr_decay = float(self.XYZ_CACHE["hr_decay"])
            del self.XYZ_CACHE["hr_decay"]

        if "hr_scaling" in self.XYZ_CACHE:
            hr_scaling = str(self.XYZ_CACHE["hr_scaling"])
            del self.XYZ_CACHE["hr_scaling"]

        KDiffusionSampler.traj_decay = hr_decay / -10.0
        KDiffusionSampler.traj_cache = None
        KDiffusionSampler.traj_scaling = hr_scaling
        KDiffusionSampler.traj_totalsteps = (
            p.steps
            if not getattr(p, "hr_second_pass_steps", 0)
            else p.hr_second_pass_steps
        )

        assert len(self.XYZ_CACHE) == 0
        return p


def restore_callback():
    KDiffusionSampler.callback_state = original_callback


script_callbacks.on_script_unloaded(restore_callback)
