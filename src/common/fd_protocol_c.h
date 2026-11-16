// the same messages as fd_protocol.hpp, for plain C (sizes must match the C++ structs)
#include <stdint.h>
#define FT_MAGIC 0x50495046u
#define FT_VERSION 2u
struct FT_Req { uint32_t magic, version, cameraMask; };
struct FT_Cam { int32_t index, bufferCount; int32_t xrFds[16]; char node[32]; };
struct FT_Reply { uint32_t magic, version; int32_t status, xrPid, cameraCount; struct FT_Cam cams[4]; };
