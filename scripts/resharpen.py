from modules.sd_samplers_kdiffusion import KDiffusionSampler
from modules import script_callbacks, shared
import modules.scripts as scripts

from scripts.res_xyz import xyz_support

from math import cos, sin, pi
import gradio as gr


def apply_scaling(alg: str, decay: float, current_step: int, total_steps: int) -> float:
    if alg == "Flat":
        return decay

    ratio = float(current_step / total_steps)
    rad = ratio * pi / 2

    match alg:
        case "Cos":
            mod = cos(rad)
        case "Sin":
            mod = sin(rad)
        case "1 - Cos":
            mod = 1 - cos(rad)
        case "1 - Sin":
            mod = 1 - sin(rad)

    return decay * mod


original_callback = KDiffusionSampler.callback_state


def hijack_callback(self, d):
    if not self.trajectory_enable:
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

            if not is_img2img:
                hr_decay = gr.Slider(
                    label="Hires. Fix Sharpness",
                    minimum=-1.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.0,
                )

            with gr.Accordion("Advanced Settings", open=False):
                scaling = gr.Radio(
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

        for comp, name in self.infotext_fields:
            comp.do_not_save_to_config = True
            self.paste_field_names.append(name)

        if not is_img2img:
            return [enable, decay, hr_decay, scaling]
        else:
            return [enable, decay, decay, scaling]

    def process(self, p, enable: bool, decay: float, hr_decay: float, scaling: str):
        if not enable:
            self.XYZ_CACHE.clear()

        if "decay" in self.XYZ_CACHE:
            decay = float(self.XYZ_CACHE["decay"])
            del self.XYZ_CACHE["decay"]

        if "scaling" in self.XYZ_CACHE:
            scaling = str(self.XYZ_CACHE["scaling"])
            if not getattr(p, "enable_hr", False):
                del self.XYZ_CACHE["scaling"]

        KDiffusionSampler.trajectory_enable = enable

        if enable:
            p.extra_generation_params["Resharpen Enabled"] = enable
            p.extra_generation_params["Resharpen Sharpness"] = decay
            p.extra_generation_params["Resharpen Scaling"] = scaling

            if p.sampler_name.strip() == "Euler a":
                print(
                    "\n[Resharpen] has little effect when using an Ancestral sampler! Consider switching to Euler instead.\n"
                )

            KDiffusionSampler.traj_decay = decay / -10.0
            KDiffusionSampler.traj_cache = None
            KDiffusionSampler.traj_scaling = scaling

            steps: int = p.steps
            # is img2img & do full steps
            if not hasattr(p, "enable_hr") and not shared.opts.img2img_fix_steps:
                if getattr(p, "denoising_strength", 1.0) < 1.0:
                    steps = int(steps * getattr(p, "denoising_strength", 1.0) + 1.0)

            KDiffusionSampler.traj_totalsteps = steps

            if getattr(p, "enable_hr", False):
                if "hr_decay" in self.XYZ_CACHE:
                    hr_decay = float(self.XYZ_CACHE["hr_decay"])
                p.extra_generation_params["Resharpen Sharpness Hires"] = hr_decay

        return p

    def before_hr(self, p, enable: bool, decay: float, hr_decay: float, scaling: str):
        if not enable:
            return p

        if "hr_decay" in self.XYZ_CACHE:
            hr_decay = float(self.XYZ_CACHE["hr_decay"])
            del self.XYZ_CACHE["hr_decay"]

        if "scaling" in self.XYZ_CACHE:
            scaling = str(self.XYZ_CACHE["scaling"])
            del self.XYZ_CACHE["scaling"]

        KDiffusionSampler.traj_decay = hr_decay / -10.0
        KDiffusionSampler.traj_cache = None
        KDiffusionSampler.traj_scaling = scaling

        return p


def restore_callback():
    KDiffusionSampler.callback_state = original_callback


script_callbacks.on_script_unloaded(restore_callback)
