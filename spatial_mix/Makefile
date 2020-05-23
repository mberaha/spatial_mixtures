ROOT_DIR := ..
PG_DIR := $(ROOT_DIR)/lib/polyagamma
PROTO_DIR := $(ROOT_DIR)/spatial_mix/protos
STAN_ROOT_DIR := /u/archive/dott/peli/usr/include/math
SRC_DIR := $(ROOT_DIR)/spatial_mix/src
SPIKES_DIR := $(SRC_DIR)/spikes
PYTHON3 := $(if $(PYTHON3), $(PYTHON3), python3)
PROTOC := /u/archive/dott/peli/usr/bin/protoc

CXX = g++
CFLAGS = \
        -std=c++1y \
        -O3 \
        -MMD \
        -I$(mkGslInc) -I/u/archive/dott/peli/usr/include/ \
        -I$(STAN_ROOT_DIR) \
        -I$(STAN_ROOT_DIR)/lib/eigen_3.3.7 \
        -I$(STAN_ROOT_DIR)/lib/boost_1.72.0 \
        -I$(STAN_ROOT_DIR)/lib/sundials_5.5.0/include \
        -I$(STAN_ROOT_DIR)/lib/tbb_2019_U8/include \
        -I$(PG_DIR) -I$(PG_DIR)/include -I$(PROTO_DIR)/cpp \
        -D_REENTRANT -fPIC

LDLIBS = $(shell pkg-config --libs protobuf)  -L$(mkGslLib) -lgsl -lgslcblas -L$(STAN_ROOT_DIR)/lib/tbb -lpthread -ltbb -Wl,-rpath,"$(STAN_ROOT_DIR)/lib/tbb" -Wl,-rpath,$(mkGslLib)

LDFLAGS = -O3 -D_REENTRANT

PROTO_SRCS = $(wildcard $(PROTO_DIR)/cpp/*.cpp)
PG_SRCS = $(wildcard $(PG_DIR)/*.cpp) $(wildcard $(PG_DIR)/include/*.cpp)
SPIKES_SRCS = $(wildcard $(SPIKES_DIR)/*.cpp)
OUR_SRCS = $(wildcard $(SRC_DIR)/*.cpp)

SRCS = $(PROTO_SRCS) $(PG_SRCS) $(OUR_SRCS)
OBJS = $(subst .cpp,.o, $(SRCS))

SPIKES_EXECS = $(subst .cpp,.out, $(SPIKES_SRCS))
SPIKES_OBJS =  $(subst .cpp,.o, $(SPIKES_SRCS))

info:
	@echo " Info..."
	@echo " ROOT_DIR  = $(ROOT_DIR)"
	@echo " PROTO_DIR = $(PROTO_DIR)"
	@echo " SRC_DIR = $(SRC_DIR)"
	@echo " SPIKES_DIR = $(SPIKES_DIR)"
	@echo " SOURCES = $(SRCS)"
	@echo " OBJECTS = $(OBJS)"
	@echo " EXECS = $(SPIKES_EXECS)"
	@echo " STAN_ROOT_DIR = $(STAN_ROOT_DIR)"
	@echo " LDLIBS = $(LDLIBS)"

all: generate_pybind

$(SPIKES_EXECS): %.out: %.o $(OBJS)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS) $< $(LDLIBS)

$(SPIKES_OBJS): %.o: %.cpp
		$(CXX) $(CFLAGS) -c $< -o $@

generate_pybind: $(OBJS)
	$(CXX) -shared $(CFLAGS) `$(PYTHON3) -m pybind11 --includes` \
		python_exports.cpp -o spmixtures`$(PYTHON3)-config --extension-suffix` \
		$(OBJS) $(LDLIBS)

%.o : %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

-include $(OBJS:%.o=%.d)

clean:
	rm $(OBJS) $(SPIKES_OBJS) run_from_file.o

distclean: clean

compile_protos:
	@ mkdir -p $(PROTO_DIR)/cpp;
	@ mkdir -p $(PROTO_DIR)/py;
	@ for filename in $(PROTO_DIR)/*.proto; do \
		$(PROTOC) --proto_path=$(PROTO_DIR) --python_out=$(PROTO_DIR)/py/ $$filename; \
		$(PROTOC) --proto_path=$(PROTO_DIR) --cpp_out=$(PROTO_DIR)/cpp/ $$filename; \
	done
	@ for filename in $(PROTO_DIR)/cpp/*.cc; do \
	    mv -- "$$filename" "$${filename%.cc}.cpp"; \
	done

	touch $(PROTO_DIR)/__init__.py
	touch $(PROTO_DIR)/py/__init__.py

	2to3 --output-dir=$(PROTO_DIR)/py/ -W -n $(PROTO_DIR)/py/
