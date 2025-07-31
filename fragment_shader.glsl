#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 objectColor;
uniform vec3 lightPos = vec3(2.0, 4.0, 2.0);
uniform vec3 lightColor = vec3(1.0, 1.0, 1.0);
uniform bool lightOn = true;

void main()
{
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 ambient = 0.1 * lightColor;
    vec3 result = (ambient + (lightOn ? diffuse : vec3(0))) * objectColor;
    FragColor = vec4(result, 1.0);
}
