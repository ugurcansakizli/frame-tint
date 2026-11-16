// grab: saves a few live frames from one of the Steam Frame's tracking cameras as .pgm pictures.
//
// ftgrab: like grab, but gets its buffer handles from the Frame Tint SteamVR driver, so it needs NO sudo:   ./ftgrab <camera index> <how many frames> <output folder>
// With <how many frames> = 0 it streams instead: every long exposure is written to standard output as a 24-byte
// header ("FTG1", width, height, frame number, time) followed by the pixels, for another program to read.
//
// How it works (the method comes from Passthrough PiP for Steam Frame, MIT licence, Copyright (c) 2026 Sippbox;
// this file reuses its approach in tracking_cameras.cpp and xrservice.hpp):
//   - XRService (Valve's tracking service) captures the cameras into shared buffers.
//   - Any program may ask the camera device which buffers exist and which one was filled last.
//   - Reading the pixels needs a copy of XRService's handle to each buffer. Passthrough PiP gets those through a
//     SteamVR add-on; this prototype takes the shortcut of running as administrator, which is allowed to copy them.
//   - The cameras alternate a long exposure (a normal picture) with a short one (nearly black, for the controller
//     lights). Only long exposures are saved.
// Nothing is written to the cameras or to XRService: buffers are mapped read-only.
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/videodev2.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include "fd_protocol_c.h"

#define MAXBUF 16

static char **g_argv; static int g_stream = 0;
// When streaming, a pause is normal (the cameras stop while nobody wears the headset, and may come back with new
// buffers). So instead of giving up, wait a second and start over from scratch. Otherwise, stop with the message.
static int problem(const char *msg) {
  int again = getenv("FTG_RETRY") != NULL;
  if (!g_stream || !again) fprintf(stderr, "%s%s\n", g_stream ? "WAITING: " : "PROBLEM: ", msg);
  if (!g_stream) return 1;
  setenv("FTG_RETRY", "1", 1);
  sleep(1);
  execv("/proc/self/exe", g_argv);
  return 1;
}
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

static int read_small(const char *path, char *buf, size_t cap) {
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return -1;
  ssize_t n = read(fd, buf, cap - 1);
  close(fd);
  if (n < 0) n = 0;
  buf[n] = 0;
  return (int)n;
}

// basename of a process's argv[0]
static void proc_name(int pid, char *out, size_t cap) {
  char path[64], cmd[512];
  out[0] = 0;
  snprintf(path, sizeof path, "/proc/%d/cmdline", pid);
  if (read_small(path, cmd, sizeof cmd) <= 0) return;
  const char *slash = strrchr(cmd, '/');
  snprintf(out, cap, "%s", slash ? slash + 1 : cmd);
}

static int parent_pid(int pid) {
  char path[64], st[512];
  snprintf(path, sizeof path, "/proc/%d/stat", pid);
  if (read_small(path, st, sizeof st) <= 0) return 0;
  const char *r = strrchr(st, ')');
  char state; int ppid = 0;
  if (!r || sscanf(r + 1, " %c %d", &state, &ppid) != 2) return 0;
  return ppid;
}

// the XRService that vrserver started
static int find_xrservice(void) {
  DIR *d = opendir("/proc");
  if (!d) return 0;
  int found = 0; struct dirent *e;
  while ((e = readdir(d))) {
    if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
    int pid = atoi(e->d_name); char name[128], pname[128];
    proc_name(pid, name, sizeof name);
    if (strcmp(name, "XRService")) continue;
    proc_name(parent_pid(pid), pname, sizeof pname);
    if (strcmp(pname, "vrserver")) continue;
    found = pid; break;
  }
  closedir(d);
  return found;
}

// XRService's log says which device each camera index uses:
//   "TrackingCameraInit: index: 0. video device: /dev/video0. ..."   (the last such line per index wins)
static int camera_node_from_log(int xr, int index, char *node, size_t cap) {
  char dir[64], link[600], target[600], logpath[600] = "";
  snprintf(dir, sizeof dir, "/proc/%d/fd", xr);
  DIR *d = opendir(dir);
  if (!d) return 0;
  struct dirent *e;
  while ((e = readdir(d))) {
    snprintf(link, sizeof link, "%s/%s", dir, e->d_name);
    ssize_t n = readlink(link, target, sizeof target - 1);
    if (n <= 4) continue;
    target[n] = 0;
    if (strstr(target, "/XRService-") && !strcmp(target + n - 4, ".log")) { snprintf(logpath, sizeof logpath, "%s", target); break; }
  }
  closedir(d);
  if (!logpath[0]) return 0;
  FILE *f = fopen(logpath, "r");
  if (!f) return 0;
  char line[2048]; int ok = 0;
  while (fgets(line, sizeof line, f)) {
    const char *at = strstr(line, "TrackingCameraInit: index: ");
    int i; char dev[64];
    if (!at || sscanf(at, "TrackingCameraInit: index: %d. video device: %63[^. \n]", &i, dev) != 2) continue;
    if (i == index && !strncmp(dev, "/dev/video", 10)) { snprintf(node, cap, "%s", dev); ok = 1; }
  }
  fclose(f);
  return ok;
}

static int query(int vfd, int i, struct v4l2_buffer *b, struct v4l2_plane *pl) {
  memset(b, 0, sizeof *b);
  memset(pl, 0, sizeof(struct v4l2_plane) * VIDEO_MAX_PLANES);
  b->type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  b->index = (uint32_t)i;
  b->m.planes = pl;
  b->length = VIDEO_MAX_PLANES;
  return ioctl(vfd, VIDIOC_QUERYBUF, b) == 0;
}

static void cpu_access(int fd, int begin) {
  struct dma_buf_sync s = {0};
  s.flags = DMA_BUF_SYNC_READ | (begin ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END);
  ioctl(fd, DMA_BUF_IOCTL_SYNC, &s);
}

static float brightness(const uint8_t *p, int w, int h, int pitch) {
  uint64_t sum = 0, cnt = 0;
  int dy = h / 32 > 0 ? h / 32 : 1, dx = w / 32 > 0 ? w / 32 : 1;
  for (int y = dy / 2; y < h; y += dy) for (int x = dx / 2; x < w; x += dx) { sum += p[(size_t)y * pitch + x]; cnt++; }
  return cnt ? (float)sum / cnt : 0.f;
}


// Asks the Frame Tint SteamVR driver (inside vrserver) for copies of the camera's buffer handles. No sudo needed.
// On success fills node, *xr, *nbuf, xrfd[] (XRService's own numbers) and ourfd[] (our copies). Returns NULL or a message.
static const char *ask_driver(int index, char *node, size_t cap, int *xr, int *nbuf, int *xrfd, int *ourfd) {
  int s = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (s < 0) return "socket failed";
  struct sockaddr_un a; memset(&a, 0, sizeof a); a.sun_family = AF_UNIX;
  snprintf(a.sun_path, sizeof a.sun_path, "/run/user/%d/frametint/cameras.sock", (int)getuid());
  if (connect(s, (struct sockaddr *)&a, sizeof a)) { close(s); return "the Frame Tint camera driver is not loaded (restart SteamVR; was it installed?)"; }
  struct FT_Req req = {FT_MAGIC, FT_VERSION, 1u << index};
  if (send(s, &req, sizeof req, MSG_NOSIGNAL) != (ssize_t)sizeof req) { close(s); return "could not talk to the camera driver"; }
  struct FT_Reply rep; memset(&rep, 0, sizeof rep);
  struct iovec io = {&rep, sizeof rep};
  union { struct cmsghdr h; char b[CMSG_SPACE(sizeof(int) * 64)]; } ctl;
  struct msghdr m; memset(&m, 0, sizeof m); m.msg_iov = &io; m.msg_iovlen = 1; m.msg_control = ctl.b; m.msg_controllen = sizeof ctl.b;
  struct timeval tv = {3, 0}; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  ssize_t n = recvmsg(s, &m, MSG_CMSG_CLOEXEC);
  close(s);
  if (n != (ssize_t)sizeof rep) return "the camera driver gave no answer";
  if (rep.status != 0) {
    switch (rep.status) {
      case 3: return "XRService has not logged its camera nodes yet (headset idle?)";
      case 4: return "the cameras have no capture buffers right now (headset idle?)";
      case 2: return "the camera driver refused this program (it must be the installed ftgrab)";
      case 1: return "the camera driver did not understand the request";
      case 5: return "the camera driver could not copy XRService's buffers";
      default: return "the camera driver reported a problem";
    }
  }
  if (rep.cameraCount != 1) return "unexpected answer from the camera driver";
  int got = 0, fds[64];
  for (struct cmsghdr *c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c))
    if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) { got = (int)((c->cmsg_len - CMSG_LEN(0)) / sizeof(int)); if (got > 64) got = 64; memcpy(fds, CMSG_DATA(c), sizeof(int) * got); }
  int bc = rep.cams[0].bufferCount;
  if (got != bc || bc <= 0 || bc > MAXBUF) { for (int i = 0; i < got; i++) close(fds[i]); return "the camera driver sent the wrong number of buffers"; }
  *xr = rep.xrPid; *nbuf = bc; snprintf(node, cap, "%s", rep.cams[0].node);
  for (int i = 0; i < bc; i++) { xrfd[i] = rep.cams[0].xrFds[i]; ourfd[i] = fds[i]; }
  return NULL;
}

int main(int argc, char **argv) {
  int index = argc > 1 ? atoi(argv[1]) : 0;
  int want = argc > 2 ? atoi(argv[2]) : 5;
  const char *outdir = argc > 3 ? argv[3] : ".";
  int every = argc > 4 ? atoi(argv[4]) : 10;         // save every Nth long exposure
  if (every < 1) every = 1;
  int stream = want == 0; g_argv = argv; g_stream = stream;
  int quiet = getenv("FTG_RETRY") != NULL;
  FILE *info = stream ? stderr : stdout;   // when streaming, standard output carries pictures only
  if (quiet) info = fopen("/dev/null", "w");

  char node[64] = ""; int xr = 0, nbuf = 0, xrfd[MAXBUF], ourfd[MAXBUF];
  { const char *err = ask_driver(index, node, sizeof node, &xr, &nbuf, xrfd, ourfd); if (err) return problem(err); }
  fprintf(info, "XRService is process %d; camera %d is %s\n", xr, index, node);

  int vfd = open(node, O_RDWR | O_NONBLOCK | O_CLOEXEC);
  if (vfd < 0) { fprintf(stderr, "PROBLEM: can't open %s: %s\n", node, strerror(errno)); return 1; }
  struct v4l2_format fmt; memset(&fmt, 0, sizeof fmt);
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  if (ioctl(vfd, VIDIOC_G_FMT, &fmt)) { fprintf(stderr, "PROBLEM: %s: VIDIOC_G_FMT: %s\n", node, strerror(errno)); return 1; }
  int w = (int)fmt.fmt.pix_mp.width, h = (int)fmt.fmt.pix_mp.height, reported = (int)fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
  fprintf(info, "picture size %dx%d, reported row length %d bytes\n", w, h, reported);
  if (w <= 0 || h <= 0 || w > 4096 || h > 4096) { fprintf(stderr, "PROBLEM: unexpected picture size\n"); return 1; }

  // which buffers exist: they must be the ones the driver copied for us (same order, same XRService handle numbers)
  size_t len[MAXBUF]; uint8_t *map[MAXBUF]; double newest = 0;
  struct v4l2_buffer b; struct v4l2_plane pl[VIDEO_MAX_PLANES];
  size_t smallest = (size_t)-1;
  for (int i = 0; i < nbuf; i++) {
    if (!query(vfd, i, &b, pl) || b.memory != V4L2_MEMORY_DMABUF || pl[0].m.fd != xrfd[i]) return problem("camera buffers do not match what the driver copied (the cameras restarted)");
    double bt = b.timestamp.tv_sec + b.timestamp.tv_usec * 1e-6;
    len[i] = pl[0].length; if (bt > newest) newest = bt;
    map[i] = mmap(NULL, len[i], PROT_READ, MAP_SHARED, ourfd[i], 0);
    if (map[i] == MAP_FAILED) { fprintf(stderr, "PROBLEM: mmap: %s\n", strerror(errno)); return 1; }
    if (len[i] < smallest) smallest = len[i];
  }
  fprintf(info, "%d shared buffers of %zu bytes each\n", nbuf, len[0]);

  // The row length the camera really writes is not always what it reports. Candidates: the width, the reported value,
  // and every multiple of 32 the buffer can hold. The right one makes neighbouring rows most alike.
  int cand[64], ncand = 0, widest = (int)(smallest / (size_t)h); if (widest > w + 512) widest = w + 512;
  if (widest < w) { fprintf(stderr, "PROBLEM: buffers too small\n"); return 1; }
  cand[ncand++] = w;
  if (reported > w && reported <= widest) cand[ncand++] = reported;
  for (int p = (w + 31) / 32 * 32; p <= widest && ncand < 64; p += 32) { int dup = 0; for (int k = 0; k < ncand; k++) dup |= cand[k] == p; if (!dup) cand[ncand++] = p; }
  double score[64] = {0}; int pitch = ncand == 1 ? cand[0] : 0, pitch_frames = 0;

  mkdir(outdir, 0755);
  uint8_t *copy = malloc((size_t)w * h);
  double last_ts = newest; uint32_t prev_seq = 0; float prev_mean = -1, recent_long = 0; int long_parity = -1, dim_run = 0;   // frames are ordered by capture time: frame numbers restart when the cameras do
  int saved = 0, longs = 0, paused = 0; double started = now_s(), last_frame = started;
  while (stream || saved < want) {
    double t = now_s();
    if (!stream && t - last_frame > 10) return problem("no new camera frames for 10 seconds (headset idle or taken off?)");
    if (stream && t - last_frame > 1.5 && !paused) { paused = 1; fprintf(stderr, "camera paused (the headset thinks nobody is wearing it); waiting...\n"); }
    // oldest not-yet-seen filled buffer that XRService holds (not queued back to the camera)
    int best = -1; uint32_t best_seq = 0; double best_ts = 0;
    for (int i = 0; i < nbuf; i++) {
      if (!query(vfd, i, &b, pl) || pl[0].m.fd != xrfd[i]) return problem("camera buffers changed (the cameras restarted)");
      double bt = b.timestamp.tv_sec + b.timestamp.tv_usec * 1e-6;
      if (!(b.flags & V4L2_BUF_FLAG_QUEUED) && bt > last_ts && (best < 0 || bt < best_ts)) { best = i; best_seq = b.sequence; best_ts = bt; }
    }
    if (best < 0) { usleep(paused ? 20000 : 1000); continue; }
    if (paused) { paused = 0; fprintf(stderr, "camera live again\n"); }
    last_ts = best_ts; last_frame = t;
    cpu_access(ourfd[best], 1);
    float mean = brightness(map[best], w, h, pitch ? pitch : cand[0]);
    // long or short exposure? compare with the frame just before
    int follows = prev_mean >= 0 && best_seq == prev_seq + 1, is_long;
    float before = prev_mean; prev_seq = best_seq; prev_mean = mean;
    if (follows && mean >= before * 1.5f + 2) { is_long = 1; long_parity = (int)(best_seq & 1); }
    else if (follows && before >= mean * 1.5f + 2) { is_long = 0; long_parity = (int)(~best_seq & 1); }
    else if (long_parity >= 0) is_long = (int)(best_seq & 1) == long_parity;
    else is_long = 0;
    if (is_long && recent_long > 0 && mean < 0.5f * recent_long && ++dim_run < 3) is_long = 0;
    if (!is_long) { cpu_access(ourfd[best], 0); continue; }
    dim_run = 0; recent_long = recent_long > 0 ? recent_long + (mean - recent_long) * 0.2f : mean;

    if (!pitch) {                                     // still working out the row length
      int rows = (int)((size_t)w * h / cand[ncand - 1]) - 1; double s[64], lo = 1e30, lo2 = 1e30;
      for (int k = 0; k < ncand; k++) {
        uint64_t diff = 0, cnt = 0;
        for (int y = 0; y + 1 < rows; y += 3) for (int x = 0; x < w; x += 3, cnt++) diff += (uint64_t)abs(map[best][(size_t)y * cand[k] + x] - map[best][(size_t)(y + 1) * cand[k] + x]);
        s[k] = cnt ? (double)diff / cnt : 0;
        if (s[k] < lo) { lo2 = lo; lo = s[k]; } else if (s[k] < lo2) lo2 = s[k];
      }
      cpu_access(ourfd[best], 0);
      if (ncand > 1 && !(lo2 > lo * 1.25 && lo2 - lo > 1.5)) continue;   // picture too plain to tell; try the next one
      for (int k = 0; k < ncand; k++) score[k] += s[k];
      if (++pitch_frames < 4) continue;
      int bk = 0; for (int k = 1; k < ncand; k++) if (score[k] < score[bk]) bk = k;
      pitch = cand[bk];
      fprintf(info, "the camera writes rows %d bytes apart\n", pitch);
      continue;
    }
    for (int y = 0; y < h; y++) memcpy(copy + (size_t)y * w, map[best] + (size_t)y * pitch, (size_t)w);
    cpu_access(ourfd[best], 0);
    // if the camera got the buffer back while we copied, the copy may be torn: skip it
    if (!query(vfd, best, &b, pl) || (b.flags & V4L2_BUF_FLAG_QUEUED) || b.sequence != best_seq) continue;
    if (longs++ % every) continue;
    if (stream) {
      struct { char magic[4]; uint32_t w, h, seq; double ts; } hd = {{'F','T','G','1'}, (uint32_t)w, (uint32_t)h, best_seq, t};
      if (fwrite(&hd, sizeof hd, 1, stdout) != 1 || fwrite(copy, 1, (size_t)w * h, stdout) != (size_t)w * h || fflush(stdout)) return 0;   // reader went away
      continue;
    }
    char path[700]; snprintf(path, sizeof path, "%s/live_%02d.pgm", outdir, saved);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "PROBLEM: can't write %s: %s\n", path, strerror(errno)); return 1; }
    fprintf(f, "P5\n%d %d\n255\n", w, h); fwrite(copy, 1, (size_t)w * h, f); fclose(f);
    printf("saved %s  (frame number %u, average brightness %.0f, %.1f s after start)\n", path, best_seq, mean, now_s() - started);
    saved++;
  }
  printf("DONE: %d live frames saved in %s\n", saved, outdir);
  return 0;
}
