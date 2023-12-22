import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_utils

#----------------------------------------------------------------------------

class ClearSkyWidget:
    def __init__(self, viz):
        self.viz        = viz
        #self.params     = dnnlib.EasyDict(phi=60.0, theta=145.0)
        self.params_def   = dnnlib.EasyDict(
            phi=62.23419205714538, # elevation
            theta=180., # azimuth
            clearsky_show=False,
            inject_clear_sky_z=False,
            inject_clear_sky_w=True,
            mask_output=True,
            normalize_azimuth=True)
        self.set_defaults()
    
    def set_defaults(self):
        self.params = dnnlib.EasyDict(self.params_def)

    def drag(self, dx, dy):
        viz = self.viz
        self.params.phi += dx / viz.font_size
        self.params.theta += dy / viz.font_size
        self._value_change()
    
    def _value_change(self):
        self.params.phi = self.params.phi % 90
        self.params.theta = self.params.theta % 360
        if self.params.normalize_azimuth:
            self.params.theta = 180.

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('Sun Position')
            imgui.same_line(viz.label_w)
            # seed = round(self.latent.x) + round(self.latent.y) * self.step_y
            # with imgui_utils.item_width(viz.font_size * 8):
            #     changed, seed = imgui.input_int('##seed', seed)
            #     if changed:
            #         self.latent.x = seed
            #         self.latent.y = 0
            imgui.same_line(viz.label_w + viz.spacing)
            with imgui_utils.item_width(viz.font_size * 5):
                changed, (new_phi, new_theta) = imgui.input_float2('##frac', self.params.phi, self.params.theta, format='%.1f', flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                if changed:
                    self.params.phi = new_phi
                    self.params.theta = new_theta
                    self._value_change()
            imgui.same_line(viz.label_w + viz.font_size * 5 + viz.spacing * 2)
            _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag', width=viz.button_w)
            if dragging:
                self.drag(dx, dy)
            imgui.same_line(viz.label_w + viz.font_size * 5 + viz.button_w + viz.spacing * 3)
            _clicked, self.params.clearsky_show = imgui.checkbox('ClearSky', self.params.clearsky_show)
            imgui.same_line(viz.label_w + viz.font_size * 5 + viz.button_w + viz.spacing * 3 + viz.button_w)
            _clicked, self.params.inject_clear_sky_z = imgui.checkbox('Inject Z', self.params.inject_clear_sky_z)
            imgui.same_line(viz.label_w + viz.font_size * 5 + viz.button_w + viz.spacing * 3 + viz.button_w * 2)
            _clicked, self.params.inject_clear_sky_w = imgui.checkbox('Inject W', self.params.inject_clear_sky_w)
            imgui.same_line(viz.label_w + viz.font_size * 5 + viz.button_w + viz.spacing * 3 + viz.button_w * 3)
            _clicked, self.params.mask_output = imgui.checkbox('Mask output', self.params.mask_output)
            # _clicked, self.latent.anim = imgui.checkbox('Anim', self.latent.anim)
            # imgui.same_line(round(viz.font_size * 27.7))
            # with imgui_utils.item_width(-1 - viz.button_w * 2 - viz.spacing * 2), imgui_utils.grayed_out(not self.latent.anim):
            #     changed, speed = imgui.slider_float('##speed', self.latent.speed, -5, 5, format='Speed %.3f', power=3)
            #     if changed:
            #         self.latent.speed = speed
            # imgui.same_line()
            # snapped = dnnlib.EasyDict(self.latent, x=round(self.latent.x), y=round(self.latent.y))
            # if imgui_utils.button('Snap', width=viz.button_w, enabled=(self.latent != snapped)):
            #     self.latent = snapped
            imgui.same_line()
            if imgui_utils.button('Reset', width=-1, enabled=(self.params != self.params_def)):
                self.set_defaults()
            
            _clicked, self.params.normalize_azimuth = imgui.checkbox('Normalize Azimuth', self.params.normalize_azimuth)
            if _clicked:
                self._value_change()

        # if self.latent.anim:
        #     self.latent.x += viz.frame_delta * self.latent.speed
        viz.args.clearsky_params = dnnlib.EasyDict(self.params)
        viz.args.clearsky_show = self.params.clearsky_show
        viz.args.inject_clear_sky_z = self.params.inject_clear_sky_z
        viz.args.inject_clear_sky_w = self.params.inject_clear_sky_w
        viz.args.mask_output = self.params.mask_output

#----------------------------------------------------------------------------
