CFLAGS = -std=c++20 -O2 -g
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi



VulkanLearning: main.cpp
	g++ $(CFLAGS) -o VulkanLearning main.cpp $(LDFLAGS)

.PHONY: test clean

test: VulkanLearning
	./VulkanLearning

clean:
	rm -f VulkanLearning
