#version 430

in vec3 fragPos;
in vec3 normal;

out vec4 FragColor;

uniform vec3 lightDir = normalize(vec3(-1.0, -1.0, -0.5));
uniform vec3 baseColor = vec3(0.8, 0.7, 0.6);

void main() {
    float diffuse = max(dot(normalize(normal), -lightDir), 0.0);
    float ambient = 0.15; // Minimum light level to avoid black
    float lighting = ambient + (1.0 - ambient) * diffuse;
    vec3 color = baseColor * lighting;
    FragColor = vec4(color, 1.0);
}
