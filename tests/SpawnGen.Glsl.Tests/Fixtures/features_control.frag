#version 300 es
precision highp float;
// SpawnGen test shader (MIT): functions and overloads, out / inout parameters (also through a swizzle), loops with break / continue,
// while and do-while, integer operators, a mutable global, consts and compound assignment.
uniform sampler2D u_image0;
uniform float u_float0;
uniform float u_float2;
uniform int u_int0;
uniform int u_int1;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

const int TAPS = 3;
const float WEIGHT = 1.0 / float(2 * TAPS + 1);
float g_total = 0.25;

float weigh(float x) { return x * WEIGHT; }
vec3 weigh(vec3 x) { return x * WEIGHT; }

void split(in vec3 c, out float lum, inout vec2 acc) {
    lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    acc += vec2(lum, 1.0);
    g_total += lum;
}

int bits(int v) {
    int count = 0;
    while (v != 0) { count += v & 1; v = v >> 1; }
    return count;
}

void main() {
    vec2 texel = 1.0 / vec2(textureSize(u_image0, 0));
    vec3 sum = vec3(0.0);
    vec2 acc = vec2(0.0);
    for (int i = -TAPS; i <= TAPS; i++) {
        if (i == 0) continue;
        vec3 s = texture(u_image0, v_texCoord + vec2(float(i), 0.0) * texel).rgb;
        sum += weigh(s);
        float lum;
        split(s, lum, acc);
        if (lum > 0.95) break;
    }
    vec4 o = vec4(sum, 1.0);
    split(o.rgb, o.a, o.xy);
    int k = 0;
    do { k++; } while (k < u_int0 + 2);
    int mask = (k << 2) | (bits(255 - u_int0 * 7) ^ 3);
    int r = (-17 % 5 + 7 % 3) * (u_int1 / 2);
    o.z = o.z * u_float0 + float(mask) * 0.01 + float(r) * 0.001;
    o.w = weigh(acc.x / max(acc.y, 1.0)) + g_total * 0.01 + u_float2 * 0.1;
    fragColor0 = o.x > 0.5 ? o : o.wzyx;
}
