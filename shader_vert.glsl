#version 430

layout(location = 0) in vec3 aPos;

uniform mat4 proj;
uniform mat4 view;

out vec3 vPos;

void main() {
    vPos = aPos;
    gl_Position = proj * view * vec4(aPos, 1.0);
}
