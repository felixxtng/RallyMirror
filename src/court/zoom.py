"""Zoom/pan view maths for the click tool (numpy only, so it can be unit tested).

The window shows a (win_w x win_h) crop of the image. `scale` is display pixels per
image pixel; (origin_x, origin_y) is the image position shown at the window's top-left.
"""
import numpy as np


class ZoomView:
    MAX_ZOOM = 12.0

    def __init__(self, img_w, img_h, win_w, win_h):
        self.img_w, self.img_h, self.win_w, self.win_h = img_w, img_h, win_w, win_h
        self.base_scale = min(win_w / img_w, win_h / img_h)
        self.reset()

    def reset(self):
        self.scale = self.base_scale
        self.origin_x = self.origin_y = 0.0

    def _clamp(self):
        """Keep the view inside the image (and centred if the image is smaller than the window)."""
        view_w, view_h = self.win_w / self.scale, self.win_h / self.scale
        self.origin_x = float(np.clip(self.origin_x, 0, max(0.0, self.img_w - view_w)))
        self.origin_y = float(np.clip(self.origin_y, 0, max(0.0, self.img_h - view_h)))

    def to_image(self, x_disp, y_disp):
        return (self.origin_x + x_disp / self.scale, self.origin_y + y_disp / self.scale)

    def to_display(self, x_img, y_img):
        return ((x_img - self.origin_x) * self.scale, (y_img - self.origin_y) * self.scale)

    def zoom_at(self, x_disp, y_disp, factor):
        """Zoom by `factor` keeping the image point under the cursor where it is."""
        anchor = self.to_image(x_disp, y_disp)
        self.scale = float(np.clip(self.scale * factor, self.base_scale, self.base_scale * self.MAX_ZOOM))
        self.origin_x = anchor[0] - x_disp / self.scale
        self.origin_y = anchor[1] - y_disp / self.scale
        self._clamp()

    def pan_by_display(self, dx_disp, dy_disp):
        """Move the view by a display-pixel amount (positive = look further right/down)."""
        self.origin_x += dx_disp / self.scale
        self.origin_y += dy_disp / self.scale
        self._clamp()

    def matrix(self):
        """2x3 affine matrix for cv2.warpAffine: image -> display."""
        return np.array([[self.scale, 0.0, -self.origin_x * self.scale],
                         [0.0, self.scale, -self.origin_y * self.scale]])
