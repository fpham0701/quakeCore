#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

static uint64_t sum_column(const std::string& csv, int col) {
  std::ifstream in(csv);
  std::string line;
  std::getline(in, line);  // header
  uint64_t sum = 0;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string tok;
    for (int i = 0; i <= col; ++i) std::getline(ss, tok, ',');
    sum += std::strtoull(tok.c_str(), nullptr, 10);
  }
  return sum;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s <map.bsp>\n", argv[0]); return 64; }
  const char* map = argv[1];

  pid_t sid = fork();
  if (sid == 0) {
    execl("./quakecore_live", "quakecore_live",
          "--transport", "tcp-listen:23457",
          "--csv", "/tmp/live.csv",
          "--threads", "4", "--block-size", "256", (char*)0);
    _exit(127);
  }
  usleep(300 * 1000);

  pid_t game = fork();
  if (game == 0) {
    execl("./fake_game", "fake_game",
          "--transport", "tcp:127.0.0.1:23457",
          "--map", map,
          "--frames", "256", "--seed", "7", (char*)0);
    _exit(127);
  }
  int st;
  waitpid(game, &st, 0); assert(WEXITSTATUS(st) == 0);
  waitpid(sid, &st, 0); assert(WEXITSTATUS(st) == 0);

  std::string cmd = std::string("./quakecore_bench --map ") + map +
                    " --frames 256 --seed 7 --threads 4 --block-size 256 --csv /tmp/bench.csv";
  int rc = std::system(cmd.c_str());
  assert(rc == 0);

  // live.csv columns: frame_id,game_ns,baseline_ns,cpu_ns,gpu_ns,game_visited,game_accepted,qc_visited,qc_accepted,drops_so_far
  // sum across all rows: column 7 = qc_visited, column 8 = qc_accepted
  uint64_t live_visited  = sum_column("/tmp/live.csv", 7);
  uint64_t live_accepted = sum_column("/tmp/live.csv", 8);

  // bench.csv columns: engine,frames,time_s,fps,visited_nodes,visited_leafs,culled_nodes,accepted_leafs
  // baseline row: column 4 = visited_nodes, column 7 = accepted_leafs
  std::ifstream bin("/tmp/bench.csv");
  std::string l;
  uint64_t b_visited = 0, b_accepted = 0;
  while (std::getline(bin, l)) {
    if (l.rfind("baseline,", 0) == 0) {
      std::stringstream ss(l);
      std::string tok;
      for (int i = 0; i < 4; ++i) std::getline(ss, tok, ',');
      std::getline(ss, tok, ','); b_visited = std::strtoull(tok.c_str(), nullptr, 10);  // col 4
      std::getline(ss, tok, ',');                                                          // col 5 visited_leafs
      std::getline(ss, tok, ',');                                                          // col 6 culled_nodes
      std::getline(ss, tok, ','); b_accepted = std::strtoull(tok.c_str(), nullptr, 10);  // col 7
      break;
    }
  }
  std::printf("live: v=%llu a=%llu  bench: v=%llu a=%llu\n",
              (unsigned long long)live_visited, (unsigned long long)live_accepted,
              (unsigned long long)b_visited, (unsigned long long)b_accepted);
  assert(live_visited == b_visited);
  assert(live_accepted == b_accepted);
  return 0;
}
