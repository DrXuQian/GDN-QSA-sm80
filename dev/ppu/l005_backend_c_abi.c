#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "gdn_qsa/ppu/backend.h"

_Static_assert(sizeof(gdn_qsa_ppu_problem_v1) == 8 * sizeof(uint32_t),
               "PPU problem ABI drifted");
_Static_assert(offsetof(gdn_qsa_ppu_problem_v1, group_chunks) == 28,
               "PPU problem field order drifted");

typedef uint64_t (*WorkspaceFn)(gdn_qsa_ppu_problem_v1 const*);
typedef int (*ForwardFn)(
    uint16_t const*, uint16_t const*, uint16_t const*, uint16_t const*,
    uint16_t const*, uint16_t*, uint16_t*, void*, uint64_t,
    gdn_qsa_ppu_problem_v1 const*, void*);

_Static_assert(
    _Generic(&gdn_qsa_ppu_workspace_size_v1, WorkspaceFn: 1, default: 0),
    "workspace query C signature drifted");
_Static_assert(
    _Generic(&gdn_qsa_ppu_forward_bf16_v1, ForwardFn: 1, default: 0),
    "forward C signature drifted");

int main(void) {
  printf("[ppu backend C ABI] %s problem_bytes=%zu group_chunks_offset=%zu\n",
         "PASS", sizeof(gdn_qsa_ppu_problem_v1),
         offsetof(gdn_qsa_ppu_problem_v1, group_chunks));
  return 0;
}
