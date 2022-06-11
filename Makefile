STB_INCLUDE_PATH = /home/user/vulkan/prompt

CFLAGS = -std=c++20 -I$(STB_INCLUDE_PATH)
app: src/main.cpp shaders/vert.spv shaders/frag.spv
	g++ $(CFLAGS) src/main.cpp -o app -lvulkan -lglfw
shaders/vert.spv: src/vert.glsl
	glslc src/vert.glsl -o shaders/vert.spv
shaders/frag.spv: src/frag.glsl
	glslc src/frag.glsl -o shaders/frag.spv
clean:
	rm app shaders/*
