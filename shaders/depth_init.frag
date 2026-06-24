#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragUV;
layout(location = 2) flat in uint fragObjectID;

layout(location = 0) out vec4 outColor;

void main() {
    // Simple coloring based on object ID
    vec3 colors[4] = vec3[](
        vec3(0.9, 0.2, 0.2),   // Red   (object 0)
        vec3(0.2, 0.4, 0.9),   // Blue  (object 1)
        vec3(0.2, 0.9, 0.3),   // Green (object 2)
        vec3(0.9, 0.8, 0.2)    // Yellow(object 3)
    );

    vec3 baseColor = colors[fragObjectID % 4];

    // Simple diffuse lighting
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    vec3 normal = normalize(fragNormal);
    float diffuse = max(dot(normal, lightDir), 0.0) * 0.7 + 0.3; // ambient = 0.3

    outColor = vec4(baseColor * diffuse, 1.0);
}