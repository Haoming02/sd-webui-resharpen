from modules.sd_samplers_kdiffusion import KDiffusionSampler
from modules import script_callbacks
import modules.scripts as scripts
import gradio as gr

original_callback = KDiffusionSampler.callback_state

def hijack_callback(self, d):
    if self.trajectory_enable is not True:
        return original_callback(self, d)

    if self.traj_cache is not None:
        delta = d['x'] - self.traj_cache
        d['x'] += delta * self.traj_decay

    self.traj_cache = d['x']
    return original_callback(self, d)

KDiffusionSampler.callback_state = hijack_callback

class ReSharpen(scripts.Script):

    def title(self):
        return "ReSharpen"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('ReSharpen', open=False):
            enable = gr.Checkbox(label="Enable")
            gr.Markdown('<h3 style="float: left;">Softer</h3> <h3 style="float: right;">Sharper</h3>')
            decay = gr.Slider(label="Sharpness", minimum=-1.0, maximum=1.0, step=0.05, value=0.0)
            if is_img2img is not True:
                hr_decay = gr.Slider(label="Hires. Fix Sharpness", minimum=-1.0, maximum=1.0, step=0.05, value=0.0)

        if is_img2img is not True:
            return [enable, decay, hr_decay]
        else:
            return [enable, decay, decay]

    def process(self, p, enable:bool, decay:float, hr_decay:float):
        KDiffusionSampler.trajectory_enable = enable

        if enable is True:
            KDiffusionSampler.traj_decay = decay / -10.0
            KDiffusionSampler.traj_cache = None

        return p

    def before_hr(self, p, enable:bool, decay:float, hr_decay:float):
        if enable is True:
            KDiffusionSampler.traj_decay = hr_decay / -10.0
            KDiffusionSampler.traj_cache = None

def restore_callback():
    KDiffusionSampler.callback_state = original_callback

script_callbacks.on_script_unloaded(restore_callback)
