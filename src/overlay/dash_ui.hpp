// The Frame Tint dashboard panel: drawn with the small software canvas (from Passthrough PiP, MIT, (c) 2026 Sippbox).
#pragma once
#include <cstdio>
#include "canvas.hpp"

namespace dashui {
constexpr int W = 800, H = 480;
constexpr float BX0 = 160, BY0 = 215, BX1 = 640, BY1 = 325;   // the big button (top-left origin)
constexpr float QX0 = 260, QY0 = 345, QX1 = 540, QY1 = 405;   // the quit button
inline bool InButton(float x, float y) { return x >= BX0 && x <= BX1 && y >= BY0 && y <= BY1; }
inline bool InQuit(float x, float y) { return x >= QX0 && x <= QX1 && y >= QY0 && y <= QY1; }
inline void Draw(framepip::Canvas &c, bool on) {
  using framepip::Color; using framepip::Align;
  c.Clear({22, 24, 30});
  c.Text(48, 92, 52, "Frame Tint", {240, 242, 246}, true);
  c.FillCircle(62, 170, 14, on ? Color{80, 210, 120} : Color{120, 124, 134});
  c.Text(92, 182, 34, on ? "ON: color view is showing" : "OFF: hidden, processing paused", {220, 224, 232});
  c.FillRoundRect(BX0, BY0, BX1, BY1, 30, on ? Color{70, 78, 96} : Color{60, 160, 100});
  c.Text((BX0 + BX1) / 2, (BY0 + BY1) / 2 + 16, 48, on ? "Switch OFF" : "Switch ON", {255, 255, 255}, true, Align::Center);
  c.FillRoundRect(QX0, QY0, QX1, QY1, 22, {120, 60, 60});
  c.Text((QX0 + QX1) / 2, (QY0 + QY1) / 2 + 11, 30, "Quit Frame Tint", {255, 255, 255}, true, Align::Center);
  c.Text(48, 455, 24, "Controller: double-tap View on the left controller (while this menu is closed)", {150, 156, 168});
}
inline void DrawThumb(framepip::Canvas &c) {            // the dock icon: three overlapping color discs
  using framepip::Color;
  c.Clear({22, 24, 30});
  c.FillCircle(48, 50, 30, {235, 80, 80}); c.FillCircle(80, 50, 30, {80, 200, 110}); c.FillCircle(64, 80, 30, {80, 130, 240});
}
}  // namespace dashui
