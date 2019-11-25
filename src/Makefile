SHELL := /bin/bash
PROTO_DIR := ./protos

CXX = g++
CFLAGS = -g -Wall -Wextra $(shell root-config --cflags)
LDLIBS = $(shell pkg-config --libs protobuf)
LDFLAGS = -g -lpthread 

PROTO_SRCS = $(wildcard $(PROTO_DIR)/cpp/*.cc)
SPIKES_SRCS = $(wildcard spikes/*.cpp)
SRCS = $(PROTO_SRCS) $(SPIKES_SRCS)

OBJS1 = $(subst .cc,.o, $(SRCS))
OBJS = $(subst .cpp,.o, $(OBJS1))

all: compile_protos; test_protos

test_protos: $(OBJS)
	$(CXX) $(LDFLAGS) -o test_protos $(OBJS)  $(LDLIBS)

%.o : %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

%.o : %.cc
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm $(OBJS)

distclean: clean

compile_protos:
	@ for filename in $(PROTO_DIR)/*.proto; do \
		protoc --proto_path=$(PROTO_DIR) --python_out=$(PROTO_DIR)/py/ $$filename; \
		protoc --proto_path=$(PROTO_DIR) --cpp_out=$(PROTO_DIR)/cpp/ $$filename; \
	done

	touch $(PROTO_DIR)/py/__init__.py

	2to3 --output-dir=$(PROTO_DIR)/py/ -W -n $(PROTO_DIR)/py/
