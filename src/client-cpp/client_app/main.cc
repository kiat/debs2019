
#include <string>
#include <iostream>
#include "Client.h"

int main() {

  // grab the host name
  const char *host = std::getenv("BENCHMARK_SYSTEM_URL");

  // ok if this env variable is not set we can not work
  if (host == nullptr) {
    return -1;
  }

  // init the client
  Client c(host);

  auto vals = c.getNextValues();

  std::vector<int> response = {12, 0, 1, 23, 0, 0};
  c.sendResponse(response);

  return 0;
}