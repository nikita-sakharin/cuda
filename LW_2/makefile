CC=nvcc
CFLAGS=-c -std=c++14 -m64 -Werror cross-execution-space-call
LDFLAGS=-m64
LIBS=-lm
SOURCES=LW_2.cu
OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=LW_2

TARGET=
ifeq ($(TARGET), Debug)
	CFLAGS+=-g
else ifeq ($(TARGET), Release)
	CFLAGS+=-O3
	LDFLAGS+=-O3
endif

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(LIBS) -o $@

$(OBJECTS): $(SOURCES)
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -frd $(OBJECTS) $(EXECUTABLE)
