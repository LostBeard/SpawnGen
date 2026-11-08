#version 300 es
precision highp float;
// SpawnGen test shader (MIT): built-in functions, constructors and swizzles, four outputs.
uniform sampler2D u_image0;
uniform vec2 u_resolution;
uniform float u_float0;
uniform float u_float1;
uniform int u_int0;
uniform bool u_bool0;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
layout(location = 1) out vec4 fragColor1;
layout(location = 2) out vec4 fragColor2;
layout(location = 3) out vec4 fragColor3;

void main() {
    vec4 c = texture(u_image0, v_texCoord);
    vec3 n = normalize(c.rgb + vec3(0.1));
    float d = dot(n, vec3(0.3, 0.6, 0.1));
    vec3 x = cross(n, vec3(0.0, 0.0, 1.0));
    vec2 p = v_texCoord * 2.0 - 1.0;
    float r = length(p);
    float a = atan(p.y, p.x);
    fragColor0 = vec4(
        smoothstep(0.2, 0.8, c.r) * step(0.3, c.g),
        mix(c.g, c.b, u_float0) + fract(r * 3.7),
        clamp(pow(abs(d), 1.7) + exp2(-r) * 0.1, 0.0, 1.0),
        distance(c.rg, vec2(u_float1)));
    fragColor1 = vec4(sin(a), cos(a * 2.0), tan(clamp(r, 0.0, 1.2)) * 0.2, mod(r * 5.0, 1.3));
    fragColor2 = vec4(reflect(vec3(p, 0.5), n), inversesqrt(r + 0.5)) * 0.5 + vec4(floor(r * 3.0), ceil(r), sign(p.x), abs(p.y)) * 0.1;
    ivec2 size = textureSize(u_image0, 0);
    vec4 texel = texelFetch(u_image0, ivec2(gl_FragCoord.xy) / 2, 0);
    bvec3 less = lessThan(c.rgb, vec3(0.5));
    float any3 = any(less) ? 1.0 : 0.0;
    float all3 = all(not(less)) ? 1.0 : 0.0;
    vec2 res = u_resolution / vec2(size);
    fragColor3 = vec4(texel.bgr * (u_bool0 ? 1.0 : 0.5), any3 * 0.5 + all3 * 0.25 + res.x * 0.1 + float(u_int0) * 0.01);
    fragColor3.xy = fragColor3.yx * vec2(1.0, 0.9) + vec2(min(c.r, c.g), max(c.b, 0.2)) * 0.1;
    fragColor3.w += log(1.0 + r) * 0.1 + sqrt(r) * 0.05 + log2(2.0 + r) * 0.01 + exp(-r) * 0.01;
}
