//
// Created by dimitrije on 2/15/19.
//



#include <iostream>
#include "Client.h"
#include "parse-float.h"

Client::Client(const std::string &baseURL) : url(baseURL) {

  // set the url for the requests
  url = baseURL;

  // init the request
  request = "GET /scene/ HTTP/1.1\r\n\r\n";

  // init the post request
  postRequest = "POST /scene/ HTTP/1.1\r\nContent-Type: application/json\r\n";

  // init the memory
  memory = std::unique_ptr<float[]> (new float[NUM_OF_READINGS]);

  // init the labels ///TODO need more stuff here
  labels = {"BigSassafras", "Oak", "ClothRecyclingContainer", "Bench", "PublicBin", "Atm", "PhoneBooth", "Cypress", "Tractor", "MotorbikeSimple", "GlassRecyclingContainer", "ToyotaPriusSimple", "MetallicTrash"};
}

int Client::socket_connect(const char *host, in_port_t port) {

  struct hostent *hp;
  struct sockaddr_in addr{};
  int on = 1, sock;

  if((hp = gethostbyname(host)) == nullptr){
    herror("gethostbyname");
    exit(1);
  }

  bcopy(hp->h_addr, &addr.sin_addr, (size_t) hp->h_length);
  addr.sin_port = htons(port);
  addr.sin_family = AF_INET;
  sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
  setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char *)&on, sizeof(int));

  if(sock == -1){
    perror("setsockopt");
    exit(1);
  }

  if(connect(sock, (struct sockaddr *)&addr, sizeof(struct sockaddr_in)) == -1){
    perror("connect");
    exit(1);

  }
  return sock;
}

float *Client::getNextValues() {

  // where to write the parsed float
  int currValue = 0;

  auto fd = socket_connect(url.c_str(), 80);
  write(fd, request.c_str(), request.size());

  int j = 0;
  char valueBuffer[1024];
  char buffer[1024];

  states s = states::init;
  while (true) {

    // read into the buffer
    ssize_t numBytes = read(fd, buffer, 1024 - 1);
    if (numBytes == 0) {
      break;
    }

    for (int i = 0; i < numBytes; ++i) {
      switch (s) {
        case states::init: {

          if (buffer[i] == '{') {
            s = read_point;
          }

          break;
        };
        case states::read_point: {

          if (buffer[i] == '{') {
            s = find_x;
          }

          break;
        };
        case states::find_x: {

          if (buffer[i] == 'X') {
            s = read_x;
          }

          break;
        }
        case states::read_x: {

          if (buffer[i] == ':') {

            i++;
            s = parse_x_val;

            for (; i < numBytes; ++i) {

              // store stuff
              if (buffer[i] != ',' && buffer[i] != '}') {
                valueBuffer[j++] = buffer[i];
              } else {

                // stop the string
                valueBuffer[j] = 0;
                char* end = nullptr;
                memory[currValue++] = fast_parse_float32(valueBuffer, &end);
                j = 0;

                s = find_y;
                break;
              }
            }
          }

          break;
        };
        case states::parse_x_val: {
          for (; i < numBytes; ++i) {

            // store stuff
            if (buffer[i] != ',' && buffer[i] != '}') {
              valueBuffer[j++] = buffer[i];
            } else {

              // stop the string
              valueBuffer[j] = 0;
              char* end = nullptr;
              memory[currValue++] = fast_parse_float32(valueBuffer, &end);
              j = 0;

              s = find_y;
              break;
            }
          }
        };
        case states::find_y: {

          if (buffer[i] == 'Y') {
            s = read_y;
          }

          break;
        }
        case states::read_y: {

          if (buffer[i] == ':') {

            i++;
            s = parse_y_val;

            for (; i < numBytes; ++i) {

              // store stuff
              if (buffer[i] != ',' && buffer[i] != '}') {
                valueBuffer[j++] = buffer[i];
              } else {

                // stop the string
                valueBuffer[j] = 0;
                char* end = nullptr;
                memory[currValue++] = fast_parse_float32(valueBuffer, &end);
                j = 0;

                s = find_z;
                break;
              }
            }
          }

          break;
        };
        case states::parse_y_val: {
          for (; i < numBytes; ++i) {

            // store stuff
            if (buffer[i] != ',' && buffer[i] != '}') {
              valueBuffer[j++] = buffer[i];
            } else {

              // stop the string
              valueBuffer[j] = 0;
              char* end = nullptr;
              memory[currValue++] = fast_parse_float32(valueBuffer, &end);
              j = 0;

              s = find_z;
              break;
            }
          }
        };
        case states::find_z: {

          if (buffer[i] == 'Z') {
            s = read_z;
          }

          break;
        }
        case states::read_z: {

          if (buffer[i] == ':') {

            i++;
            s = parse_z_val;

            for (; i < numBytes; ++i) {

              // store stuff
              if (buffer[i] != ',' && buffer[i] != '}') {
                valueBuffer[j++] = buffer[i];
              } else {

                // stop the string
                valueBuffer[j] = 0;
                char* end = nullptr;
                memory[currValue++] = fast_parse_float32(valueBuffer, &end);
                j = 0;

                s = read_point;
                break;
              }
            }
          }

          break;
        };
        case states::parse_z_val: {
          for (; i < numBytes; ++i) {

            // store stuff
            if (buffer[i] != ',' && buffer[i] != '}') {
              valueBuffer[j++] = buffer[i];
            } else {

              // stop the string
              valueBuffer[j] = 0;
              char* end = nullptr;
              memory[currValue++] = fast_parse_float32(valueBuffer, &end);
              j = 0;

              s = read_point;
              break;
            }
          }
        };
      };
    }

    buffer[numBytes] = 0;
  }

  shutdown(fd, SHUT_RDWR);
  close(fd);

  return memory.get();
}

void Client::sendResponse(const std::vector<int> &out) {

  std::string output = "{";
  output.reserve(1024);

  for(int i = 0; i < out.size(); ++i) {

    if(out[i] != 0) {
      output += "'";
      output += labels[i];
      output += "':'";
      output += std::to_string(out[i]);
      output += "',";
    }
  }

  // remove the last comma if needed
  if(output.size() > 1) {
    output.pop_back();
  }

  output += "}";

  // write the request
  auto fd = socket_connect(url.c_str(), 80);

  std::string prelude = "Content-Length: " + std::to_string(output.size()) + "\r\n\r\n";
  std::string sd = postRequest + prelude + output;

  // write the json
  write(fd, sd.c_str(), sd.size());

  // skip the response
  char buffer[1024];
  while (true) {

    ssize_t numBytes = read(fd, buffer, 1024 - 1);
    if (numBytes == 0) {
      break;
    }

    buffer[numBytes] =0;

    std::cout << buffer;

  }

  // shutdown
  shutdown(fd, SHUT_RDWR);
  close(fd);
}
