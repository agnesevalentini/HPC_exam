
# Compilers for wsl
CC = gcc
MPICC = mpicc
ARCH ?= native

CFLAGS = -Iinclude -Wall -march=$(ARCH)
# The flags 
MPI_FLAGS = -Wall -Iinclude -fopenmp -O3 -march=$(ARCH)
SRC_DIR = src
SERIAL_SRC = $(SRC_DIR)/stencil_template_serial.c
PARALLEL_SRC = $(SRC_DIR)/stencil_template_parallel.c
SERIAL_OUT = stencil_serial
PARALLEL_OUT = stencil_parallel

# Targets
all: $(PARALLEL_OUT)
serial: $(SERIAL_OUT)
parallel: $(PARALLEL_OUT)

$(SERIAL_OUT): $(SERIAL_SRC)
	$(CC) $(CFLAGS) $(SERIAL_SRC) -o $(SERIAL_OUT)

$(PARALLEL_OUT): $(PARALLEL_SRC)
	$(MPICC) $(MPI_FLAGS) $(PARALLEL_SRC) -o $(PARALLEL_OUT) -lm

#run-serial: $(SERIAL_OUT)
#	./$(SERIAL_OUT) -f 0 -e 4 -o 0 -p 1 -x 100 -y 100 -n 5

#run-parallel: $(PARALLEL_OUT)
#	mpirun -n 4 ./$(PARALLEL_OUT) -e 4 -o 1 -p 0 -x 100 -y 100 -n 5 -v 1

clean:
	rm -f $(SERIAL_OUT) $(PARALLEL_OUT) *.o