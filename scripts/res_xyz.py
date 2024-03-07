import modules.scripts as scripts


def grid_reference():
    for data in scripts.scripts_data:
        if data.script_class.__module__ == 'xyz_grid.py' and hasattr(data, "module"):
            return data.module

    raise SystemError("Could not find X/Y/Z Plot...")


def xyz_support(cache: dict):

    def apply_field(field):
        def _(p, x, xs):
            cache.update({field : x})
        return _

    xyz_grid = grid_reference()

    extra_axis_options = [
        xyz_grid.AxisOption("[ReSharpen] Sharpness", float, apply_field("decay")),
        xyz_grid.AxisOption("[ReSharpen] HrF. Sharpness", float, apply_field("hr_decay"))
    ]

    xyz_grid.axis_options.extend(extra_axis_options)
