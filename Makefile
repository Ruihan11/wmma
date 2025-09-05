NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_89 -std=c++17 -lcuda -lcudart -lcublas
TARGET = wmmaBenchmark
KERNEL_DIR = kernels
KERNEL_SOURCES = $(wildcard $(KERNEL_DIR)/*.cu)
SOURCES = wmmaBenchmark.cu $(KERNEL_SOURCES)

.PHONY: all clean run plot

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) *.csv *.png

run: $(TARGET)
	./$(TARGET)

test: $(TARGET)
	./$(TARGET) 512 512 512

plot: $(TARGET)
	./$(TARGET)
	python3 plot_results.py

list:
	@echo "Available kernel files:"
	@ls -1 $(KERNEL_DIR)/*.cu | sed 's|$(KERNEL_DIR)/||'

help:
	@echo "Available targets:"
	@echo "  all          - Build the benchmark"
	@echo "  clean        - Remove built files"
	@echo "  run          - Run benchmark with default matrix sizes"
	@echo "  test         - Run benchmark with 512x512x512 matrices"
	@echo "  plot         - Run benchmark and create performance plot"
	@echo "  list         - List all kernel files"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Kernel Files:"
	@ls -1 $(KERNEL_DIR)/*.cu | sed 's|$(KERNEL_DIR)/||' | sed 's/^/  /'
	@echo ""
	@echo "Usage: ./$(TARGET) [M] [N] [K] [iterations]"
	@echo "  M, N, K     - Matrix dimensions (optional, defaults to multiple sizes)"
	@echo "  iterations  - Number of benchmark iterations (optional, default: 100)"
	@echo ""
	@echo "Examples:"
	@echo "  ./$(TARGET)              - Test all default sizes with 100 iterations"
	@echo "  ./$(TARGET) 1024 1024 1024 200 - Test 1024³ matrices with 200 iterations"