#version 430

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 vPos[];

out vec3 fragPos;
out vec3 normal;

void main() {
    vec3 p0 = vPos[0];
    vec3 p1 = vPos[1];
    vec3 p2 = vPos[2];

    vec3 n = normalize(cross(p1 - p0, p2 - p0)); // Face normal

    for (int i = 0; i < 3; ++i) {
        fragPos = vPos[i];
        normal = n;
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
