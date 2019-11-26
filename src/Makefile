PROTO_DIR := ./protos
PG_DIR := ../lib/polyagamma

CXX = g++
CFLAGS = \
	-std=c++1y \
	-g \
	-I../lib/math \
	-I/usr/include/eigen3/ \
	-I../lib/math/lib/boost_1.69.0/  \
	-I../lib/math/lib/sundials_4.1.0/include  \
	-I../lib/math/lib/tbb_2019_U8/include  \
	-I ../lib/polyagamma -I ../lib/polyagamma/include \
	-D_REENTRANT $(shell root-config --cflags)
LDLIBS = \
 	$(shell pkg-config --libs protobuf) -L../lib/math/lib/tbb \
	-lgsl -lgslcblas -lpthread -ltbb -Wl,-rpath,"../lib/math/lib/tbb"
LDFLAGS = -g -D_REENTRANT

PROTO_SRCS = $(wildcard $(PROTO_DIR)/cpp/*.cpp)
PG_SRCS = $(wildcard $(PG_DIR)/*.cpp) $(wildcard $(PG_DIR)/include/*.cpp)
SPIKES_SRCS = $(wildcard spikes/*.cpp)
OUR_SRCS = $(wildcard *.cpp)

SRCS = $(PROTO_SRCS) $(PG_SRCS) $(OUR_SRCS)
OBJS = $(subst .cpp,.o, $(SRCS))

SPIKES_EXECS = $(subst .cpp,.out, $(SPIKES_SRCS))

info:
	@echo " Info..."
	@echo " SOURCES = $(SRCS)"
	@echo " OBJECTS = $(OBJS)"
	@echo " EXECS = $(EXECS)"

all: compile_protos test_protos.out test_stan.out test_pg.out test_sampler.out test_normal_gamma.out

test_protos.out: test_protos.o $(OBJS)
	$(CXX) $(LDFLAGS) -o test_protos.out $(OBJS) test_protos.o  $(LDLIBS)

test_protos.o:
	$(CXX) $(CFLAGS) -c spikes/test_protos.cpp -o test_protos.o

test_stan.out: test_stan.o $(OBJS)
	$(CXX) $(LDFLAGS) -o test_stan.out $(OBJS) test_stan.o  $(LDLIBS)

test_stan.o:
	$(CXX) $(CFLAGS) -c spikes/test_stan.cpp -o test_stan.o

test_pg.out: test_pg.o $(OBJS)
	$(CXX) $(LDFLAGS) -o test_pg.out $(OBJS) test_pg.o  $(LDLIBS)

test_pg.o:
	$(CXX) $(CFLAGS) -c spikes/test_pg.cpp -o test_pg.o

test_sampler.out: test_sampler.o $(OBJS)
	$(CXX) $(LDFLAGS) -o test_sampler.out $(OBJS) test_sampler.o  $(LDLIBS)

test_sampler.o:
	$(CXX) $(CFLAGS) -c spikes/test_sampler.cpp -o test_sampler.o

test_normal_gamma.out: test_normal_gamma.o $(OBJS)
	$(CXX) $(LDFLAGS) -o test_normal_gamma.out $(OBJS) test_normal_gamma.o  $(LDLIBS)

test_normal_gamma.o:
	$(CXX) $(CFLAGS) -c spikes/test_normal_gamma.cpp -o test_normal_gamma.o


%.o : %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

-include $(dep)

%.d: %.c
	$(CXX) $(CFLAGS) $< -MM -MT $(@:.d=.o) >$@

clean:
	rm $(OBJS) test_stan.o test_protos.o

distclean: clean

compile_protos:
	@ for filename in $(PROTO_DIR)/*.proto; do \
		protoc --proto_path=$(PROTO_DIR) --python_out=$(PROTO_DIR)/py/ $$filename; \
		protoc --proto_path=$(PROTO_DIR) --cpp_out=$(PROTO_DIR)/cpp/ $$filename; \
	done
	@ for filename in protos/cpp/*.cc; do \
	    mv -- "$$filename" "$${filename%.cc}.cpp"; \
	done

	touch $(PROTO_DIR)/py/__init__.py

	2to3 --output-dir=$(PROTO_DIR)/py/ -W -n $(PROTO_DIR)/py/
