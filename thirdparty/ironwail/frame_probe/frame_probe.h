#ifndef IRONWAIL_FRAME_PROBE_H
#define IRONWAIL_FRAME_PROBE_H

#include <stdint.h>

void FrameProbe_Init(const char* transport_spec, int handshake_once);
void FrameProbe_Shutdown(void);

/* Called inside the renderer immediately around the cull pass. */
void FrameProbe_BeginCull(void);
void FrameProbe_EndCull(uint32_t game_visited_nodes, uint32_t game_accepted_leafs);

/* Called after EndCull, just before R_DrawWorld. Emits a FramePacket. */
void FrameProbe_Emit(void);

#endif
