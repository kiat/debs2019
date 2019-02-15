//
// Created by dimitrije on 2/15/19.
//

#ifndef HELLO_CLIENT_H
#define HELLO_CLIENT_H

#include <string>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>
#include <memory>
#include <vector>

const size_t NUM_OF_READINGS = 72000 * 3;

enum states {

  init,
  read_point,
  find_x,
  read_x,
  parse_x_val,
  find_y,
  read_y,
  parse_y_val,
  find_z,
  read_z,
  parse_z_val
};

class Client {
public:

  explicit Client(const std::string &baseURL);

  float* getNextValues();

  void sendResponse(const std::vector<int> &out);

 private:

  int socket_connect(const char *host, in_port_t port);

  std::unique_ptr<float[]> memory;

  std::string url;

  std::string request;

  std::string postRequest;

  std::vector<std::string> labels;
};

#endif //HELLO_CLIENT_H
