#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 fragColor;

layout(push_constant) uniform constants {
    mat4 modelViewProj;
} pushConstants;

void main() {
    gl_Position = pushConstants.modelViewProj * vec4(inPosition, 1.0);
    fragColor = inColor;
}