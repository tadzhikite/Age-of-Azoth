#version 450
#pragma shader_stage(vertex)
layout( push_constant ) uniform ColorBlock {
      mat4 masdf;
      uint phase;
} p;
layout(binding = 0) uniform UniformBufferObject {
    mat4 mode[24];
} ubo;

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inCoord;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    gl_Position = p.masdf * ubo.mode[p.phase] * inPosition;
    fragColor = inColor;
    fragTexCoord = inCoord;
}
