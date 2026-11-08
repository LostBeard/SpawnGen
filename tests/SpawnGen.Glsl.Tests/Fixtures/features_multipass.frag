#version 300 es
#pragma passes 3
precision highp float;
// SpawnGen test shader (MIT): three passes; each later pass reads the one before through u_image0.
uniform sampler2D u_image0;
uniform int u_pass;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(u_image0, 0));
    vec2 dir = u_pass == 0 ? vec2(1.0, 0.0) : u_pass == 1 ? vec2(0.0, 1.0) : vec2(0.7, 0.7);
    vec4 c = texture(u_image0, v_texCoord) * 0.5;
    c += texture(u_image0, v_texCoord + dir * texel * 1.5) * 0.25;
    c += texture(u_image0, v_texCoord - dir * texel * 1.5) * 0.25;
    fragColor0 = u_pass == 2 ? vec4(c.rgb, 1.0) : c;
}
