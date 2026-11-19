// ftoverlay: shows pictures inside the Steam Frame as a panel that stays in front of your eyes.
//
//   ./ftoverlay <panel width in metres> <distance in metres> <opacity 0..1> [poses] [sbs]
//
// With the word  sbs  each picture holds two views side by side: the left half is shown to the left eye and the
// right half to the right eye. FT_TEXEL (default 0.5) corrects the panel's shape for that double-width picture.
//
// With the word  poses  it also writes where the head is, about 100 times a second, to standard output: "FTP1" then
// 12 numbers (SteamVR's 3x4 head matrix, room coordinates). The color memory uses that to pin colors to the room.
//
// It reads pictures from standard input: a 12-byte header ("FTV1", width, height) followed by width*height*4 bytes
// of RGBA, repeated. Each picture replaces the last one on the panel. It ends when the input ends.
// A picture may instead start with "FTV2", width, height, a flag and the 3x4 head matrix the picture belongs to: the
// panel is then pinned in the room where the head was at that moment, so between pictures SteamVR keeps it lined up
// with the room at the display's full speed instead of dragging it along with the head.
//
// A picture may also start with "FTS1": like FTV2, but instead of the pixels it carries one number, which of the
// pictures in the shared memory file named by FT_SHM to show. That avoids pushing every pixel through the pipe.
// With  poses , a 52-byte "FTA1" note is written to standard output after each picture has been handed to SteamVR.
// With  poses gaze , a 52-byte "FTE1" record follows each head pose: float valid, then where the eyes look as a
// direction relative to the head (x/-z, y/-z; x right, y up), from SteamVR's eye-tracked foveation centre.
//
// Pictures are handed to SteamVR as graphics-card textures, two of them used in turn, so the panel never shows a
// missing or half-updated picture (which is what made the first version flicker).
//
// The way of joining SteamVR (as a background app) and the texture upload follow Passthrough PiP for Steam Frame,
// MIT licence, Copyright (c) 2026 Sippbox.
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <libgen.h>
#include <sstream>
#include <thread>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "openvr.h"
#include "dash_ui.hpp"

static bool read_all(void *dst, size_t n) { return fread(dst, 1, n, stdin) == n; }

struct Gpu {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t family = 0;
  VkPhysicalDeviceMemoryProperties mem{};
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMem = VK_NULL_HANDLE;
  void *mapped = nullptr;
  VkImage images[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  VkDeviceMemory imageMem[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  int w = 0, h = 0, next = 0;
  vr::VRVulkanTextureData_t tex{};

  static std::vector<std::string> Words(const std::string &s) {
    std::vector<std::string> out;
    std::istringstream in(s);
    std::string w;
    while (in >> w) out.push_back(w);
    return out;
  }
  uint32_t MemType(uint32_t bits, VkMemoryPropertyFlags props) const {
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props) return i;
    return UINT32_MAX;
  }
  const char *Init() {
    vr::IVRCompositor *comp = vr::VRCompositor();
    std::vector<std::string> want;
    if (comp) {
      uint32_t n = comp->GetVulkanInstanceExtensionsRequired(nullptr, 0);
      if (n) { std::string s(n, '\0'); comp->GetVulkanInstanceExtensionsRequired(&s[0], n); want = Words(s.c_str()); }
    }
    uint32_t an = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &an, nullptr);
    std::vector<VkExtensionProperties> avail(an);
    vkEnumerateInstanceExtensionProperties(nullptr, &an, avail.data());
    std::vector<const char *> on;
    for (const std::string &e : want)
      for (const VkExtensionProperties &a : avail)
        if (e == a.extensionName) on.push_back(a.extensionName);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "frametint";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = (uint32_t)on.size();
    ici.ppEnabledExtensionNames = on.data();
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) return "vkCreateInstance failed";

    uint64_t hmd = 0;
    if (vr::VRSystem()) vr::VRSystem()->GetOutputDevice(&hmd, vr::TextureType_Vulkan, instance);
    physical = (VkPhysicalDevice)hmd;
    if (!physical) {
      uint32_t n = 1;
      if (vkEnumeratePhysicalDevices(instance, &n, &physical) < 0 || !n) return "no graphics device";
    }
    vkGetPhysicalDeviceMemoryProperties(physical, &mem);
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qn, qf.data());
    bool found = false;
    for (uint32_t i = 0; i < qn && !found; i++)
      if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) family = i, found = true;
    if (!found) return "no graphics queue";

    std::vector<std::string> dwant;
    if (comp) {
      uint32_t n = comp->GetVulkanDeviceExtensionsRequired(physical, nullptr, 0);
      if (n) { std::string s(n, '\0'); comp->GetVulkanDeviceExtensionsRequired(physical, &s[0], n); dwant = Words(s.c_str()); }
    }
    uint32_t dn = 0;
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &dn, nullptr);
    std::vector<VkExtensionProperties> davail(dn);
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &dn, davail.data());
    std::vector<const char *> don;
    for (const std::string &e : dwant)
      for (const VkExtensionProperties &a : davail)
        if (e == a.extensionName) don.push_back(a.extensionName);
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)don.size();
    dci.ppEnabledExtensionNames = don.data();
    if (vkCreateDevice(physical, &dci, nullptr, &device) != VK_SUCCESS) return "vkCreateDevice failed";
    vkGetDeviceQueue(device, family, 0, &queue);

    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp.queueFamilyIndex = family;
    if (vkCreateCommandPool(device, &cp, nullptr, &pool) != VK_SUCCESS) return "command pool failed";
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &ca, &cmd) != VK_SUCCESS) return "command buffer failed";
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(device, &fci, nullptr, &fence) != VK_SUCCESS) return "fence failed";
    return nullptr;
  }
  bool Size(int width, int height) {
    if (width == w && height == h) return true;
    if (w) return false;                                   // the picture size is not expected to change
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = (VkDeviceSize)width * height * 4;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateBuffer(device, &bci, nullptr, &staging) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, staging, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = MemType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(device, &mai, nullptr, &stagingMem) != VK_SUCCESS) return false;
    vkBindBufferMemory(device, staging, stagingMem, 0);
    if (vkMapMemory(device, stagingMem, 0, bci.size, 0, &mapped) != VK_SUCCESS) return false;
    for (int i = 0; i < 2; i++) {
      VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      ici.imageType = VK_IMAGE_TYPE_2D;
      ici.format = VK_FORMAT_R8G8B8A8_UNORM;
      ici.extent = {(uint32_t)width, (uint32_t)height, 1};
      ici.mipLevels = 1;
      ici.arrayLayers = 1;
      ici.samples = VK_SAMPLE_COUNT_1_BIT;
      ici.tiling = VK_IMAGE_TILING_OPTIMAL;
      ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      if (vkCreateImage(device, &ici, nullptr, &images[i]) != VK_SUCCESS) return false;
      vkGetImageMemoryRequirements(device, images[i], &req);
      mai.allocationSize = req.size;
      mai.memoryTypeIndex = MemType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(device, &mai, nullptr, &imageMem[i]) != VK_SUCCESS) return false;
      vkBindImageMemory(device, images[i], imageMem[i], 0);
    }
    w = width; h = height;
    return true;
  }
  void Barrier(VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags sa, VkAccessFlags da, VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcAccessMask = sa;
    b.dstAccessMask = da;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
  }
  bool Upload(const uint8_t *rgba, int width, int height, vr::Texture_t *out) {
    if (!Size(width, height)) return false;
    memcpy(mapped, rgba, (size_t)width * height * 4);
    VkImage img = images[next];
    next ^= 1;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(cmd, 0);
    vkBeginCommandBuffer(cmd, &bi);
    Barrier(img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {(uint32_t)width, (uint32_t)height, 1};
    vkCmdCopyBufferToImage(cmd, staging, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    Barrier(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkResetFences(device, 1, &fence);
    if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS || vkWaitForFences(device, 1, &fence, VK_TRUE, 1000000000ull) != VK_SUCCESS) return false;
    tex.m_nImage = (uint64_t)img;
    tex.m_pDevice = device;
    tex.m_pPhysicalDevice = physical;
    tex.m_pInstance = instance;
    tex.m_pQueue = queue;
    tex.m_nQueueFamilyIndex = family;
    tex.m_nWidth = (uint32_t)width;
    tex.m_nHeight = (uint32_t)height;
    tex.m_nFormat = VK_FORMAT_R8G8B8A8_UNORM;
    tex.m_nSampleCount = 1;
    out->handle = &tex;
    out->eType = vr::TextureType_Vulkan;
    out->eColorSpace = vr::ColorSpace_Gamma;
    return true;
  }
};

static std::atomic<bool> g_run{true};

// Where the eyes look, as a direction relative to the head: x/-z and y/-z (x right, y up, -z forward).
// SteamVR gives the per-eye foveation centre in each eye's NDC; undo that eye's projection and its eye-to-head turn,
// then average the two eyes. Returns false while the eye tracker has no answer (blink, headset off).
struct GazeReader {
  float p00[2], p02[2], p11[2], p12[2];
  vr::HmdMatrix34_t eyeToHead[2];
  GazeReader() {
    for (int e = 0; e < 2; e++) {
      const vr::HmdMatrix44_t P = vr::VRSystem()->GetProjectionMatrix((vr::EVREye)e, 0.1f, 100.f);
      p00[e] = P.m[0][0]; p02[e] = P.m[0][2]; p11[e] = P.m[1][1]; p12[e] = P.m[1][2];
      eyeToHead[e] = vr::VRSystem()->GetEyeToHeadTransform((vr::EVREye)e);
    }
  }
  bool Read(float *tx, float *ty) const {
    vr::HmdVector2_t ndc[2];
    if (!vr::VRSystem()->GetEyeTrackedFoveationCenter(&ndc[0], &ndc[1])) return false;
    float s[3] = {0, 0, 0};
    for (int e = 0; e < 2; e++) {
      const float v[3] = {(ndc[e].v[0] + p02[e]) / p00[e], (ndc[e].v[1] + p12[e]) / p11[e], -1.f};
      float h[3];
      for (int r = 0; r < 3; r++) h[r] = eyeToHead[e].m[r][0] * v[0] + eyeToHead[e].m[r][1] * v[1] + eyeToHead[e].m[r][2] * v[2];
      const float n = sqrtf(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
      for (int r = 0; r < 3; r++) s[r] += h[r] / n;
    }
    if (s[2] > -0.1f) return false;
    *tx = s[0] / -s[2];
    *ty = s[1] / -s[2];
    return true;
  }
};

static void PoseLoop(bool gaze) {
  GazeReader *gr = gaze ? new GazeReader() : nullptr;
  if (gr) fprintf(stderr, "overlay: sending where the eyes look (gaze)\n");
  while (g_run) {
    vr::TrackedDevicePose_t p{};
    vr::VRSystem()->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, &p, 1);
    if (p.bPoseIsValid) {
      char rec[52];
      memcpy(rec, "FTP1", 4);
      memcpy(rec + 4, &p.mDeviceToAbsoluteTracking.m[0][0], 48);
      if (write(1, rec, sizeof rec) != (ssize_t)sizeof rec) break;
    }
    if (gr) {                                              // "FTE1", float valid, float x/-z, float y/-z (head directions)
      char rec[52] = {0};
      float g[3] = {0, 0, 0};
      g[0] = gr->Read(&g[1], &g[2]) ? 1.f : 0.f;
      memcpy(rec, "FTE1", 4);
      memcpy(rec + 4, g, sizeof g);
      if (write(1, rec, sizeof rec) != (ssize_t)sizeof rec) break;
    }
    usleep(10000);
  }
  delete gr;
}

static std::atomic<bool> g_on{true};                       // false while the user has switched Frame Tint off from the controller
static bool g_notes = false;                               // write "FTT1" notes to the program that sends pictures
static std::atomic<bool> g_dirty{true};                    // the dashboard panel needs redrawing
static void SetOn(vr::VROverlayHandle_t h, bool on, const char *how) {
  g_on = on; g_dirty = true;
  if (on) vr::VROverlay()->ShowOverlay(h); else vr::VROverlay()->HideOverlay(h);
  fprintf(stderr, "overlay: Frame Tint switched %s from %s\n", on ? "ON" : "OFF", how);
  if (g_notes) { char rec[52] = {0}; memcpy(rec, "FTT1", 4); float f = on ? 1.f : 0.f; memcpy(rec + 4, &f, 4); if (write(1, rec, sizeof rec) != (ssize_t)sizeof rec) return; }
}
// A panel in the SteamVR dashboard with a big on/off button (works while the dashboard is open, unlike controller bindings).
static void DashLoop(vr::VROverlayHandle_t view, std::string dir) {
  const std::string fontDir = dir + "/fonts";   // dir = the share folder
  vr::IVROverlay *ov = vr::VROverlay();
  vr::VROverlayHandle_t dash = vr::k_ulOverlayHandleInvalid, thumb = vr::k_ulOverlayHandleInvalid;
  vr::EVROverlayError oe = ov->CreateDashboardOverlay(getenv("FT_DASH_KEY") ? getenv("FT_DASH_KEY") : "frametint.dashboard.v2", "Frame Tint", &dash, &thumb);
  if (oe != vr::VROverlayError_None) { fprintf(stderr, "overlay: no dashboard panel (%s)\n", ov->GetOverlayErrorNameFromEnum(oe)); return; }
  ov->SetOverlayWidthInMeters(dash, 2.3f);
  ov->SetOverlayInputMethod(dash, vr::VROverlayInputMethod_Mouse);
  vr::HmdVector2_t scale = {{(float)dashui::W, (float)dashui::H}};
  ov->SetOverlayMouseScale(dash, &scale);
  framepip::Canvas canvas(dashui::W, dashui::H), icon(128, 128);
  std::string err;
  canvas.LoadFonts(fontDir, &err);                         // only the text fonts matter here
  dashui::DrawThumb(icon);
  vr::EVROverlayError te = ov->SetOverlayFromFile(thumb, (dir + "/icon.png").c_str());
  fprintf(stderr, "overlay: dock icon from %s/icon.png: %s\n", dir.c_str(), ov->GetOverlayErrorNameFromEnum(te));
  { uint32_t tw = 0, th = 0; vr::EVROverlayError ze = ov->GetOverlayTextureSize(thumb, &tw, &th);
    fprintf(stderr, "overlay: SteamVR holds the dock icon as %ux%u (%s)\n", tw, th, ov->GetOverlayErrorNameFromEnum(ze)); }
  if (te != vr::VROverlayError_None) {
    te = ov->SetOverlayRaw(thumb, (void *)icon.Pixels(), 128, 128, 4);
    fprintf(stderr, "overlay: dock icon from memory: %s\n", ov->GetOverlayErrorNameFromEnum(te));
  }
  auto reapplyIcon = [&](const char *why) {
    vr::EVROverlayError e2 = ov->SetOverlayFromFile(thumb, (dir + "/icon.png").c_str());
    fprintf(stderr, "overlay: dock icon again (%s): %s\n", why, ov->GetOverlayErrorNameFromEnum(e2));
  };
  fprintf(stderr, "overlay: dock handles: panel %llu, icon %llu\n", (unsigned long long)dash, (unsigned long long)thumb);
  fprintf(stderr, "overlay: dashboard panel ready\n");
  const auto t0 = std::chrono::steady_clock::now(); bool late = false;
  while (g_run) {
    vr::VREvent_t ev;
    while (ov->PollNextOverlayEvent(dash, &ev, sizeof ev)) {
      static int seen[8], nseen = 0; bool known = false;   // log each kind of panel event once, to see whether SteamVR talks to the panel
      for (int i = 0; i < nseen; i++) known |= seen[i] == (int)ev.eventType;
      if (!known && nseen < 8) { seen[nseen++] = (int)ev.eventType; fprintf(stderr, "overlay: dashboard panel event %d (%s)\n", (int)ev.eventType, vr::VRSystem()->GetEventTypeNameFromEnum((vr::EVREventType)ev.eventType)); }
      if (ev.eventType == vr::VREvent_OverlayShown || ev.eventType == vr::VREvent_FocusEnter) reapplyIcon("panel shown");
      if (ev.eventType == vr::VREvent_MouseButtonDown && dashui::InButton(ev.data.mouse.x, dashui::H - ev.data.mouse.y))
        SetOn(view, !g_on.load(), "the dashboard");
      if (ev.eventType == vr::VREvent_MouseButtonDown && dashui::InQuit(ev.data.mouse.x, dashui::H - ev.data.mouse.y)) {
        fprintf(stderr, "overlay: Quit pressed in the dashboard\n");
        vr::VROverlay()->HideOverlay(view);
        if (g_notes) { char rec[52] = {0}; memcpy(rec, "FTQ1", 4); if (write(1, rec, sizeof rec) != (ssize_t)sizeof rec) _exit(0); }   // the main program ends everything
        else _exit(0);
      }
    }
    if (!late && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(4)) { late = true; reapplyIcon("4 s after start"); }
    if (g_dirty.exchange(false)) {
      dashui::Draw(canvas, g_on.load());
      vr::EVROverlayError re = ov->SetOverlayRaw(dash, (void *)canvas.Pixels(), dashui::W, dashui::H, 4);
      static bool said = false; if (!said) { said = true; fprintf(stderr, "overlay: dashboard panel picture: %s\n", ov->GetOverlayErrorNameFromEnum(re)); }
    }
    usleep(20000);
  }
  ov->DestroyOverlay(dash);
}
// Double-tap View on the left controller (default binding, changeable in SteamVR's controller settings) shows / hides the picture.
static void InputLoop(vr::VROverlayHandle_t h, std::string manifest) {
  vr::IVRInput *in = vr::VRInput();
  if (!in) { fprintf(stderr, "overlay: SteamVR Input unavailable; no toggle button\n"); return; }
  vr::VRActionSetHandle_t set = vr::k_ulInvalidActionSetHandle;
  vr::VRActionHandle_t act = vr::k_ulInvalidActionHandle;
  vr::EVRInputError e = in->SetActionManifestPath(manifest.c_str());
  if (e == vr::VRInputError_None) e = in->GetActionSetHandle("/actions/frametint", &set);
  if (e == vr::VRInputError_None) e = in->GetActionHandle("/actions/frametint/in/toggle", &act);
  if (e != vr::VRInputError_None) { fprintf(stderr, "overlay: toggle button not available (SteamVR Input error %d, %s)\n", (int)e, manifest.c_str()); return; }
  fprintf(stderr, "overlay: toggle button ready (double-tap View on the left controller)\n");
  bool down = false;
  while (g_run) {
    vr::VRActiveActionSet_t a{};
    a.ulActionSet = set;
    in->UpdateActionState(&a, sizeof a, 1);
    vr::InputDigitalActionData_t d{};
    const bool now = in->GetDigitalActionData(act, &d, sizeof d, vr::k_ulInvalidInputValueHandle) == vr::VRInputError_None && d.bActive && d.bState;
    if (now && !down) SetOn(h, !g_on.load(), "the controller");
    down = now;
    usleep(16000);
  }
}

int main(int argc, char **argv) {
  const float width = argc > 1 ? (float)atof(argv[1]) : 2.3f;
  const float dist = argc > 2 ? (float)atof(argv[2]) : 1.5f;
  const float alpha = argc > 3 ? (float)atof(argv[3]) : 1.0f;

  vr::EVRInitError e = vr::VRInitError_None;
  for (int tries = 0; tries < 15; tries++) {
    vr::VR_Init(&e, vr::VRApplication_Background);
    if (e == vr::VRInitError_None) break;
    sleep(2);
  }
  if (e != vr::VRInitError_None) {
    fprintf(stderr, "PROBLEM (overlay): could not join SteamVR: %s\n", vr::VR_GetVRInitErrorAsEnglishDescription(e));
    return 1;
  }
  vr::IVROverlay *ov = vr::VROverlay();
  vr::VROverlayHandle_t h = vr::k_ulOverlayHandleInvalid;
  vr::EVROverlayError oe = ov ? ov->CreateOverlay("frametint.view", "Frame Tint view", &h) : vr::VROverlayError_RequestFailed;
  if (oe == vr::VROverlayError_KeyInUse && ov) oe = ov->FindOverlay("frametint.view", &h);   // left over from a previous run
  if (oe != vr::VROverlayError_None) {
    fprintf(stderr, "PROBLEM (overlay): could not create the panel: %s\n", ov ? ov->GetOverlayErrorNameFromEnum(oe) : "no overlay interface");
    vr::VR_Shutdown();
    return 1;
  }
  ov->SetOverlayWidthInMeters(h, width);
  vr::HmdMatrix34_t m = {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, -dist}}};   // straight ahead of the eyes, facing them
  ov->SetOverlayTransformTrackedDeviceRelative(h, vr::k_unTrackedDeviceIndex_Hmd, &m);
  ov->SetOverlayAlpha(h, alpha);

  Gpu gpu;
  const char *gerr = gpu.Init();
  bool useGpu = gerr == nullptr;
  fprintf(stderr, "overlay: panel ready (%.2f m wide, %.2f m away, opacity %.2f), %s\n", width, dist, alpha,
          useGpu ? "smooth texture path" : "FALLBACK path that may flicker");
  if (gerr) fprintf(stderr, "overlay: graphics setup said: %s\n", gerr);

  bool wantPoses = false, sbs = false, wantGaze = false;
  for (int i = 4; i < argc; i++) {
    if (!strcmp(argv[i], "poses")) wantPoses = true;
    if (!strcmp(argv[i], "sbs")) sbs = true;
    if (!strcmp(argv[i], "gaze")) wantGaze = true;
  }
  if (sbs) {
    const float texel = getenv("FT_TEXEL") ? (float)atof(getenv("FT_TEXEL")) : 0.5f;
    vr::EVROverlayError fe = ov->SetOverlayFlag(h, vr::VROverlayFlags_SideBySide_Parallel, true);
    if (texel > 0) ov->SetOverlayTexelAspect(h, texel);
    fprintf(stderr, "overlay: one view per eye (side by side): SteamVR said %s; texel shape %.2f\n", ov->GetOverlayErrorNameFromEnum(fe), texel);
  }
  std::thread poseThread;
  if (wantPoses) poseThread = std::thread(PoseLoop, wantGaze);
  g_notes = wantPoses;
  std::thread inputThread, dashThread;
  {
    char exe[4096]; ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n > 0) {
      exe[n] = 0;
      const std::string bin = dirname(exe);              // (dirname edits its argument, so only call it once)
      std::string dir = getenv("FT_SHARE") ? getenv("FT_SHARE") : bin + "/../share";   // input/, fonts/ and icon.png live here
      { char real[4096]; if (realpath(dir.c_str(), real)) dir = real; }
      std::string manifest = dir + "/input/actions.json";
      if (access(manifest.c_str(), R_OK) == 0) inputThread = std::thread(InputLoop, h, manifest);
      else fprintf(stderr, "overlay: no %s; no toggle button\n", manifest.c_str());
      dashThread = std::thread(DashLoop, h, dir);
    }
  }

  const uint8_t *shm = nullptr;
  size_t shmSize = 0;
  if (getenv("FT_SHM")) {
    int fd = open(getenv("FT_SHM"), O_RDONLY | O_CLOEXEC);
    struct stat st;
    if (fd >= 0 && fstat(fd, &st) == 0 && st.st_size > 0) {
      void *mp = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
      if (mp != MAP_FAILED) { shm = (const uint8_t *)mp; shmSize = (size_t)st.st_size; }
    }
    if (fd >= 0) close(fd);
    fprintf(stderr, shm ? "overlay: pictures come through shared memory\n" : "overlay: could not open the shared memory file\n");
  }
  std::vector<uint8_t> pixels;
  bool shown = false;
  long frames = 0;
  int lastErr = -1;
  for (;;) {
    char magic[4];
    uint32_t wh[2];
    if (!read_all(magic, 4) || !read_all(wh, 8)) break;
    const bool s1 = !memcmp(magic, "FTS1", 4);
    const bool v2 = s1 || !memcmp(magic, "FTV2", 4);
    uint32_t hasPose = 0, which = 0;
    float hp[12];
    if (v2 && (!read_all(&hasPose, 4) || !read_all(hp, 48))) break;
    if (s1 && !read_all(&which, 4)) break;
    if ((!v2 && memcmp(magic, "FTV1", 4)) || wh[0] == 0 || wh[1] == 0 || wh[0] > 4096 || wh[1] > 4096) {
      fprintf(stderr, "PROBLEM (overlay): unexpected picture data\n");
      break;
    }
    const size_t bytes = (size_t)wh[0] * wh[1] * 4;
    const uint8_t *px;
    if (s1) {
      if (!shm || ((size_t)which + 1) * bytes > shmSize) { fprintf(stderr, "PROBLEM (overlay): shared picture %u is not there\n", which); break; }
      px = shm + (size_t)which * bytes;
    } else {
      pixels.resize(bytes);
      if (!read_all(pixels.data(), pixels.size())) break;
      px = pixels.data();
    }

    vr::VREvent_t ev;                                       // keep SteamVR's message queue empty; leave when asked
    bool quit = false;
    while (vr::VRSystem()->PollNextEvent(&ev, sizeof ev))
      if (ev.eventType == vr::VREvent_Quit) quit = true;
    if (quit) { fprintf(stderr, "overlay: SteamVR asked us to close\n"); break; }

    if (v2 && hasPose) {                                    // panel = head-at-that-moment, then `dist` straight ahead
      vr::HmdMatrix34_t a;
      for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) a.m[r][c] = hp[r * 4 + c];
        a.m[r][3] = hp[r * 4 + 3] - hp[r * 4 + 2] * dist;
      }
      ov->SetOverlayTransformAbsolute(h, vr::TrackingUniverseStanding, &a);
    } else {
      ov->SetOverlayTransformTrackedDeviceRelative(h, vr::k_unTrackedDeviceIndex_Hmd, &m);
    }
    vr::Texture_t tex{};
    if (useGpu && gpu.Upload(px, (int)wh[0], (int)wh[1], &tex)) {
      oe = ov->SetOverlayTexture(h, &tex);
    } else {
      if (useGpu) { fprintf(stderr, "overlay: texture upload failed, switching to the fallback path\n"); useGpu = false; }
      oe = ov->SetOverlayRaw(h, (void *)px, wh[0], wh[1], 4);
    }
    if ((int)oe != lastErr) {
      fprintf(stderr, "overlay: picture %ld: SteamVR said %s\n", frames, ov->GetOverlayErrorNameFromEnum(oe));
      lastErr = (int)oe;
    }
    if (g_on && (!shown || !ov->IsOverlayVisible(h))) {
      ov->ShowOverlay(h);
      fprintf(stderr, shown ? "overlay: panel had been hidden at picture %ld, showing it again\n" : "overlay: showing pictures in the headset\n", frames);
      shown = true;
    }
    frames++;
    if (wantPoses) {                                        // tell the sender this picture is on its way to the display
      char note[52] = {0};
      memcpy(note, "FTA1", 4);
      if (write(1, note, sizeof note) != (ssize_t)sizeof note) break;
    }
  }
  fprintf(stderr, "overlay: stopped after %ld pictures\n", frames);
  g_run = false;
  if (poseThread.joinable()) poseThread.join();
  if (inputThread.joinable()) inputThread.join();
  if (dashThread.joinable()) dashThread.join();
  ov->HideOverlay(h);
  ov->DestroyOverlay(h);
  vr::VR_Shutdown();
  return 0;
}
