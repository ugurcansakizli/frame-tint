// gaze_probe: can a SteamVR background app (like ftoverlay) read where the eyes look on the Steam Frame?
//
//   g++ -O2 -std=c++17 -I<repo>/third_party/openvr tools/gaze_probe.cpp -o gaze_probe -L/opt/steamvr/bin/linuxarm64 -lopenvr_api -Wl,-rpath,/opt/steamvr/bin/linuxarm64
//   ./gaze_probe [seconds] [binding variant 0|1]
//
// Tries three routes at once, 5 times a second, and prints the gaze direction relative to the head
// (yaw: + is right, pitch: + is up, in degrees):
//   A  IVRSystem::GetEyeTrackedFoveationCenter (per-eye foveation point; no setup needed)
//   B  an "eyetracking" SteamVR Input action bound to /user/eyes_ext/input/gaze_ext (variant 1: .../gaze_ext/pose)
//   C  the HMD property Prop_SupportsXrEyeGazeInteraction_Bool (printed once)
// Nothing is stored or sent anywhere; it only prints. Ends with RESULT lines.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <string>

#include "openvr.h"

static bool WriteFile(const std::string &path, const std::string &text) {
  FILE *f = fopen(path.c_str(), "w");
  if (!f) return false;
  fputs(text.c_str(), f);
  fclose(f);
  return true;
}

// direction (room coordinates) -> yaw/pitch relative to the head pose m
static void HeadAngles(const vr::HmdMatrix34_t &m, const float d[3], float *yaw, float *pitch) {
  float h[3];  // head = R^T * d
  for (int c = 0; c < 3; c++) h[c] = m.m[0][c] * d[0] + m.m[1][c] * d[1] + m.m[2][c] * d[2];
  *yaw = atan2f(h[0], -h[2]) * 57.2958f;
  *pitch = atan2f(h[1], sqrtf(h[0] * h[0] + h[2] * h[2])) * 57.2958f;
}

struct Stats {
  int asked = 0, ok = 0;
  float ymin = 1e9f, ymax = -1e9f, pmin = 1e9f, pmax = -1e9f;
  void Add(float y, float p) {
    ok++;
    ymin = fminf(ymin, y); ymax = fmaxf(ymax, y); pmin = fminf(pmin, p); pmax = fmaxf(pmax, p);
  }
  void Print(const char *name) const {
    if (!ok) { printf("RESULT %s: no gaze in %d tries\n", name, asked); return; }
    printf("RESULT %s: gaze in %d of %d tries; yaw %.1f..%.1f deg, pitch %.1f..%.1f deg (moved %.1f x %.1f deg)\n", name, ok, asked,
           ymin, ymax, pmin, pmax, ymax - ymin, pmax - pmin);
  }
};

int main(int argc, char **argv) {
  const int seconds = argc > 1 ? atoi(argv[1]) : 20;
  const int variant = argc > 2 ? atoi(argv[2]) : 0;

  vr::EVRInitError e = vr::VRInitError_None;
  vr::VR_Init(&e, vr::VRApplication_Background);
  if (e != vr::VRInitError_None) {
    printf("RESULT init: could not join SteamVR: %s\n", vr::VR_GetVRInitErrorAsEnglishDescription(e));
    return 1;
  }
  vr::IVRSystem *sys = vr::VRSystem();

  // C: does the headset say it supports eye gaze?
  vr::ETrackedPropertyError pe = vr::TrackedProp_Success;
  const bool sup = sys->GetBoolTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SupportsXrEyeGazeInteraction_Bool, &pe);
  printf("RESULT property: SupportsXrEyeGazeInteraction = %s (%s)\n", sup ? "true" : "false", sys->GetPropErrorNameFromEnum(pe));

  // B: an eyetracking action with a default binding for the Frame headset
  const std::string dir = "/tmp/frametint-gaze-probe";
  (void)!system(("mkdir -p " + dir).c_str());
  const char *path = variant == 1 ? "/user/eyes_ext/input/gaze_ext/pose" : "/user/eyes_ext/input/gaze_ext";
  WriteFile(dir + "/actions.json",
            "{\n \"default_bindings\": [ { \"controller_type\": \"frame_hmd\", \"binding_url\": \"bindings_frame_hmd.json\" } ],\n"
            " \"actions\": [ { \"name\": \"/actions/gazeprobe/in/gaze\", \"type\": \"eyetracking\" } ],\n"
            " \"action_sets\": [ { \"name\": \"/actions/gazeprobe\", \"usage\": \"leftright\" } ],\n"
            " \"localization\": [ { \"language_tag\": \"en_US\", \"/actions/gazeprobe\": \"Gaze probe\", \"/actions/gazeprobe/in/gaze\": \"Gaze\" } ]\n}\n");
  WriteFile(dir + "/bindings_frame_hmd.json",
            std::string("{\n \"action_manifest_version\": 0, \"controller_type\": \"frame_hmd\", \"name\": \"gaze probe\", \"category\": \"steamvr_input\",\n"
                        " \"bindings\": { \"/actions/gazeprobe\": { \"eyetracking\": [ { \"path\": \"") + path +
                "\", \"output\": \"/actions/gazeprobe/in/gaze\" } ] } }\n}\n");
  vr::IVRInput *in = vr::VRInput();
  vr::VRActionSetHandle_t set = vr::k_ulInvalidActionSetHandle;
  vr::VRActionHandle_t act = vr::k_ulInvalidActionHandle;
  vr::EVRInputError ie = in ? in->SetActionManifestPath((dir + "/actions.json").c_str()) : vr::VRInputError_InvalidHandle;
  if (ie == vr::VRInputError_None) ie = in->GetActionSetHandle("/actions/gazeprobe", &set);
  if (ie == vr::VRInputError_None) ie = in->GetActionHandle("/actions/gazeprobe/in/gaze", &act);
  printf("input action setup (binding %s): error %d\n", path, (int)ie);

  float l, r, t, b;
  sys->GetProjectionRaw(vr::Eye_Right, &l, &r, &t, &b);
  { float l2, r2, t2, b2; sys->GetProjectionRaw(vr::Eye_Left, &l2, &r2, &t2, &b2);
    printf("projection raw (tangents) right eye: left %.3f right %.3f top %.3f bottom %.3f | left eye: %.3f %.3f %.3f %.3f\n", l, r, t, b, l2, r2, t2, b2); }

  Stats A, B;
  int lastBErr = -999;
  printf("look left, right, up and down for %d seconds...\n", seconds);
  for (int i = 0; i < seconds * 5; i++) {
    vr::TrackedDevicePose_t hp{};
    sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, &hp, 1);
    const vr::HmdMatrix34_t &m = hp.mDeviceToAbsoluteTracking;
    char line[256] = "";

    // A: foveation centre of the right eye (NDC) -> angles through the right eye's projection
    vr::HmdVector2_t nl{}, nr{};
    A.asked++;
    if (sys->GetEyeTrackedFoveationCenter(&nl, &nr)) {
      // invert the eye's projection matrix (NDC y is up): view tangent = (ndc + m[i][2]) / m[i][i]
      const vr::HmdMatrix44_t P = sys->GetProjectionMatrix(vr::Eye_Right, 0.1f, 100.f);
      const float tx = (nr.v[0] + P.m[0][2]) / P.m[0][0], ty = (nr.v[1] + P.m[1][2]) / P.m[1][1];
      const float yaw = atanf(tx) * 57.2958f, pitch = atanf(ty) * 57.2958f;
      A.Add(yaw, pitch);
      snprintf(line + strlen(line), sizeof line - strlen(line), "A yaw %6.1f pitch %6.1f (ndc L %.2f,%.2f R %.2f,%.2f) | ", yaw, pitch, nl.v[0], nl.v[1], nr.v[0], nr.v[1]);
    } else {
      snprintf(line + strlen(line), sizeof line - strlen(line), "A none              | ");
    }

    // B: the input action
    B.asked++;
    if (act != vr::k_ulInvalidActionHandle) {
      vr::VRActiveActionSet_t as{};
      as.ulActionSet = set;
      in->UpdateActionState(&as, sizeof as, 1);
      vr::VREyeTrackingData_t g{};
      const vr::EVRInputError ge = in->GetEyeTrackingDataRelativeToNow(act, vr::TrackingUniverseStanding, 0.f, &g, sizeof g);
      if ((int)ge != lastBErr) { printf("B: GetEyeTrackingDataRelativeToNow error %d\n", (int)ge); lastBErr = (int)ge; }
      if (ge == vr::VRInputError_None && g.bValid && hp.bPoseIsValid) {
        float d[3] = {g.vGazeTarget.v[0] - g.vGazeOrigin.v[0], g.vGazeTarget.v[1] - g.vGazeOrigin.v[1], g.vGazeTarget.v[2] - g.vGazeOrigin.v[2]};
        float yaw, pitch;
        HeadAngles(m, d, &yaw, &pitch);
        B.Add(yaw, pitch);
        snprintf(line + strlen(line), sizeof line - strlen(line), "B yaw %6.1f pitch %6.1f (active %d tracked %d)", yaw, pitch, g.bActive, g.bTracked);
      } else {
        snprintf(line + strlen(line), sizeof line - strlen(line), "B none (active %d valid %d)", g.bActive, g.bValid);
      }
    }
    if (i % 5 == 0) printf("%5.1f s  %s\n", i / 5.0, line);
    usleep(200000);
  }
  A.Print("A foveation centre");
  B.Print("B eyetracking action");
  vr::VR_Shutdown();
  return 0;
}
