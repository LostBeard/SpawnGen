// A small glTF binary (.glb) viewer: orbit with the left button, pan with the right, zoom with the wheel; double-click fits the model again.
// Reads positions, normals, UVs, indices and the base colour texture of every mesh (the layout TRELLIS.2 and most exporters write).
// No libraries: one WebGL2 shader (a hemisphere light and a key light over the base colour).
(function () {
    const TYPE_SIZES = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
    const ARRAYS = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };

    function parseGlb(buffer) {
        const view = new DataView(buffer);
        if (buffer.byteLength < 12 || view.getUint32(0, true) !== 0x46546C67) throw new Error("This is not a .glb file.");
        let offset = 12, json = null, bin = null;
        while (offset + 8 <= view.byteLength) {
            const length = view.getUint32(offset, true), type = view.getUint32(offset + 4, true);
            const start = offset + 8;
            if (type === 0x4E4F534A) json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, start, length)));
            else if (type === 0x004E4942 && !bin) bin = buffer.slice(start, start + length);
            offset = start + length + ((4 - length % 4) % 4);
        }
        if (!json) throw new Error("The .glb has no JSON chunk.");
        return { json, bin };
    }

    // The values of an accessor as a plain typed array (de-interleaved), plus its component count.
    function readAccessor(glb, index) {
        const acc = glb.json.accessors[index];
        const Array_ = ARRAYS[acc.componentType], n = TYPE_SIZES[acc.type];
        const out = new Array_(acc.count * n);
        if (acc.bufferView === undefined) return { data: out, n, acc };
        const bv = glb.json.bufferViews[acc.bufferView];
        const base = (bv.byteOffset || 0) + (acc.byteOffset || 0);
        const elementSize = Array_.BYTES_PER_ELEMENT * n, stride = bv.byteStride || elementSize;
        const src = new DataView(glb.bin);
        const getter = { 5120: "getInt8", 5121: "getUint8", 5122: "getInt16", 5123: "getUint16", 5125: "getUint32", 5126: "getFloat32" }[acc.componentType];
        for (let i = 0; i < acc.count; i++)
            for (let c = 0; c < n; c++)
                out[i * n + c] = src[getter](base + i * stride + c * Array_.BYTES_PER_ELEMENT, true);
        return { data: out, n, acc };
    }

    function matrixOf(node) {
        if (node.matrix) return Float32Array.from(node.matrix);
        const t = node.translation || [0, 0, 0], q = node.rotation || [0, 0, 0, 1], s = node.scale || [1, 1, 1];
        const [x, y, z, w] = q;
        return Float32Array.from([
            (1 - 2 * (y * y + z * z)) * s[0], (2 * (x * y + z * w)) * s[0], (2 * (x * z - y * w)) * s[0], 0,
            (2 * (x * y - z * w)) * s[1], (1 - 2 * (x * x + z * z)) * s[1], (2 * (y * z + x * w)) * s[1], 0,
            (2 * (x * z + y * w)) * s[2], (2 * (y * z - x * w)) * s[2], (1 - 2 * (x * x + y * y)) * s[2], 0,
            t[0], t[1], t[2], 1]);
    }

    function mul(a, b) {
        const r = new Float32Array(16);
        for (let c = 0; c < 4; c++) for (let rr = 0; rr < 4; rr++)
            r[c * 4 + rr] = a[rr] * b[c * 4] + a[4 + rr] * b[c * 4 + 1] + a[8 + rr] * b[c * 4 + 2] + a[12 + rr] * b[c * 4 + 3];
        return r;
    }

    function perspective(fovy, aspect, near, far) {
        const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
        return Float32Array.from([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
    }

    function lookAt(eye, target) {
        const up = [0, 1, 0];
        let zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
        const zl = Math.hypot(zx, zy, zz) || 1; zx /= zl; zy /= zl; zz /= zl;
        let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
        const xl = Math.hypot(xx, xy, xz) || 1; xx /= xl; xy /= xl; xz /= xl;
        const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
        return Float32Array.from([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
            -(xx * eye[0] + xy * eye[1] + xz * eye[2]), -(yx * eye[0] + yy * eye[1] + yz * eye[2]), -(zx * eye[0] + zy * eye[1] + zz * eye[2]), 1]);
    }

    const VERTEX = `#version 300 es
        in vec3 aPos; in vec3 aNormal; in vec2 aUv;
        uniform mat4 uModel; uniform mat4 uView; uniform mat4 uProj;
        out vec3 vNormal; out vec2 vUv;
        void main() {
            vNormal = mat3(uView * uModel) * aNormal;
            vUv = aUv;
            gl_Position = uProj * uView * uModel * vec4(aPos, 1.0);
        }`;
    const FRAGMENT = `#version 300 es
        precision mediump float;
        in vec3 vNormal; in vec2 vUv;
        uniform sampler2D uTex; uniform vec4 uBase; uniform float uHasTex; uniform float uHasNormals;
        out vec4 outColor;
        void main() {
            vec4 base = uBase;
            if (uHasTex > 0.5) base *= texture(uTex, vUv);
            vec3 n = normalize(vNormal);
            if (!gl_FrontFacing) n = -n;
            float light = 1.0;
            if (uHasNormals > 0.5) {
                float key = max(dot(n, normalize(vec3(0.4, 0.6, 0.7))), 0.0);
                float fill = max(dot(n, normalize(vec3(-0.5, 0.1, 0.4))), 0.0);
                float sky = 0.5 + 0.5 * n.y;
                light = 0.5 + 0.3 * sky + 0.55 * key + 0.2 * fill;
            }
            outColor = vec4(pow(base.rgb * light, vec3(1.0)), base.a);
        }`;

    function compile(gl, type, source) {
        const s = gl.createShader(type);
        gl.shaderSource(s, source); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        return s;
    }

    // Builds everything the GPU needs for the model: one draw item per mesh primitive, and the bounds of the whole thing.
    async function load(gl, glb) {
        const items = [];
        const textures = new Map();
        const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];

        async function textureFor(textureIndex) {
            if (textureIndex === undefined) return null;
            if (textures.has(textureIndex)) return textures.get(textureIndex);
            const tex = glb.json.textures[textureIndex];
            const image = glb.json.images[tex.source];
            let bitmap = null;
            try {
                if (image.bufferView !== undefined) {
                    const bv = glb.json.bufferViews[image.bufferView];
                    const bytes = new Uint8Array(glb.bin, bv.byteOffset || 0, bv.byteLength);
                    bitmap = await createImageBitmap(new Blob([bytes], { type: image.mimeType || "image/png" }), { imageOrientation: "none", premultiplyAlpha: "none", colorSpaceConversion: "none" });
                }
            } catch (e) { bitmap = null; }
            let glTex = null;
            if (bitmap) {
                glTex = gl.createTexture();
                gl.bindTexture(gl.TEXTURE_2D, glTex);
                gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
                gl.generateMipmap(gl.TEXTURE_2D);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, tex.sampler !== undefined ? (glb.json.samplers?.[tex.sampler]?.wrapS ?? gl.REPEAT) : gl.REPEAT);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, tex.sampler !== undefined ? (glb.json.samplers?.[tex.sampler]?.wrapT ?? gl.REPEAT) : gl.REPEAT);
            }
            textures.set(textureIndex, glTex);
            return glTex;
        }

        async function visit(nodeIndex, parent) {
            const node = glb.json.nodes[nodeIndex];
            const world = mul(parent, matrixOf(node));
            if (node.mesh !== undefined) {
                for (const prim of glb.json.meshes[node.mesh].primitives) {
                    if (prim.mode !== undefined && prim.mode !== 4) continue; // triangles only
                    const pos = readAccessor(glb, prim.attributes.POSITION);
                    const nrm = prim.attributes.NORMAL !== undefined ? readAccessor(glb, prim.attributes.NORMAL) : null;
                    const uv = prim.attributes.TEXCOORD_0 !== undefined ? readAccessor(glb, prim.attributes.TEXCOORD_0) : null;
                    const idx = prim.indices !== undefined ? readAccessor(glb, prim.indices).data : null;
                    const count = idx ? idx.length : pos.acc.count;
                    if (!idx) throw new Error("This viewer needs indexed triangles.");
                    const material = prim.material !== undefined ? glb.json.materials[prim.material] : {};
                    const pbr = material.pbrMetallicRoughness || {};
                    const make = (data, size) => {
                        const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); return { buffer: b, size };
                    };
                    const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
                    const indexData = idx instanceof Uint32Array || idx instanceof Uint16Array ? idx : Uint32Array.from(idx);
                    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indexData, gl.STATIC_DRAW);
                    for (let i = 0; i < pos.acc.count; i++) {
                        const x = pos.data[i * 3], y = pos.data[i * 3 + 1], z = pos.data[i * 3 + 2];
                        const wx = world[0] * x + world[4] * y + world[8] * z + world[12], wy = world[1] * x + world[5] * y + world[9] * z + world[13], wz = world[2] * x + world[6] * y + world[10] * z + world[14];
                        if (wx < min[0]) min[0] = wx; if (wy < min[1]) min[1] = wy; if (wz < min[2]) min[2] = wz;
                        if (wx > max[0]) max[0] = wx; if (wy > max[1]) max[1] = wy; if (wz > max[2]) max[2] = wz;
                    }
                    items.push({
                        pos: make(pos.data, 3), normal: nrm ? make(nrm.data, 3) : null, uv: uv ? make(uv.data, 2) : null,
                        indices: ib, count, indexType: indexData instanceof Uint16Array ? gl.UNSIGNED_SHORT : gl.UNSIGNED_INT, model: world,
                        base: pbr.baseColorFactor || [1, 1, 1, 1], texture: await textureFor(pbr.baseColorTexture?.index), doubleSided: !!material.doubleSided,
                    });
                }
            }
            for (const child of node.children || []) await visit(child, world);
        }

        const scene = glb.json.scenes?.[glb.json.scene ?? 0];
        const roots = scene ? scene.nodes : (glb.json.nodes || []).map((_, i) => i);
        const identity = Float32Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
        for (const root of roots) await visit(root, identity);
        if (!items.length) throw new Error("The .glb holds no triangles to show.");
        return { items, min, max };
    }

    // Opens the viewer over the page. Resolves when it is closed. `url` is anything fetch can read (a data URL, or a same-origin file).
    window.spawngenView3D = function (url, title) {
        return new Promise(async resolve => {
            const root = document.createElement("div");
            root.className = "glbviewer";
            root.innerHTML = '<div class="gv-bar"><span class="gv-title"></span><span class="gv-hint">drag: orbit · right drag: pan · wheel: zoom · double-click: fit</span>' +
                '<span class="gv-spacer"></span><button class="gv-download">Download</button><button class="gv-close">Close</button></div>' +
                '<canvas class="gv-canvas"></canvas><div class="gv-status">Loading...</div>';
            root.querySelector(".gv-title").textContent = title || "3D model";
            document.body.appendChild(root);
            const canvas = root.querySelector(".gv-canvas"), status = root.querySelector(".gv-status");
            let raf = 0, disposed = false;
            const finish = () => {
                if (disposed) return;
                disposed = true;
                cancelAnimationFrame(raf);
                document.removeEventListener("keydown", onKey);
                root.remove();
                resolve();
            };
            const onKey = e => { if (e.key === "Escape") { e.preventDefault(); finish(); } };
            document.addEventListener("keydown", onKey);
            root.querySelector(".gv-close").onclick = finish;
            root.querySelector(".gv-download").onclick = () => {
                const a = document.createElement("a"); a.href = url; a.download = (title || "model") + ".glb"; document.body.appendChild(a); a.click(); a.remove();
            };
            try {
                const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
                if (!gl) throw new Error("This browser has no WebGL2.");
                const glb = parseGlb(await (await fetch(url)).arrayBuffer());
                const model = await load(gl, glb);
                const program = gl.createProgram();
                gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX));
                gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
                gl.linkProgram(program);
                if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
                const loc = { pos: gl.getAttribLocation(program, "aPos"), normal: gl.getAttribLocation(program, "aNormal"), uv: gl.getAttribLocation(program, "aUv") };
                const u = n => gl.getUniformLocation(program, n);
                const uniforms = { model: u("uModel"), view: u("uView"), proj: u("uProj"), tex: u("uTex"), base: u("uBase"), hasTex: u("uHasTex"), hasNormals: u("uHasNormals") };
                const centre = [0, 1, 2].map(i => (model.min[i] + model.max[i]) / 2);
                const radius = Math.max(1e-6, Math.hypot(model.max[0] - model.min[0], model.max[1] - model.min[1], model.max[2] - model.min[2]) / 2);
                let yaw = 0.6, pitch = 0.25, dist = radius * 2.6, target = centre.slice();
                const fit = () => { yaw = 0.6; pitch = 0.25; dist = radius * 2.6; target = centre.slice(); };

                let dragging = null;
                canvas.oncontextmenu = e => e.preventDefault();
                canvas.onpointerdown = e => { dragging = { x: e.clientX, y: e.clientY, button: e.button }; canvas.setPointerCapture(e.pointerId); };
                canvas.onpointermove = e => {
                    if (!dragging) return;
                    const dx = e.clientX - dragging.x, dy = e.clientY - dragging.y;
                    dragging.x = e.clientX; dragging.y = e.clientY;
                    if (dragging.button === 0) { yaw -= dx * 0.008; pitch = Math.max(-1.5, Math.min(1.5, pitch + dy * 0.008)); }
                    else {
                        // Pan across the view plane.
                        const k = dist * 0.0015;
                        const rx = [Math.cos(yaw), 0, -Math.sin(yaw)];
                        target[0] -= rx[0] * dx * k; target[2] -= rx[2] * dx * k; target[1] += dy * k;
                    }
                };
                canvas.onpointerup = canvas.onpointercancel = () => { dragging = null; };
                canvas.onwheel = e => { e.preventDefault(); dist = Math.max(radius * 0.2, Math.min(radius * 30, dist * Math.exp(e.deltaY * 0.001))); };
                canvas.ondblclick = fit;

                status.textContent = model.items.length + " mesh" + (model.items.length === 1 ? "" : "es");
                const draw = () => {
                    if (disposed) return;
                    const ratio = window.devicePixelRatio || 1;
                    const w = Math.max(1, Math.floor(canvas.clientWidth * ratio)), h = Math.max(1, Math.floor(canvas.clientHeight * ratio));
                    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
                    gl.viewport(0, 0, w, h);
                    gl.clearColor(0.13, 0.13, 0.14, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
                    gl.enable(gl.DEPTH_TEST);
                    gl.useProgram(program);
                    const eye = [target[0] + dist * Math.cos(pitch) * Math.sin(yaw), target[1] + dist * Math.sin(pitch), target[2] + dist * Math.cos(pitch) * Math.cos(yaw)];
                    gl.uniformMatrix4fv(uniforms.view, false, lookAt(eye, target));
                    gl.uniformMatrix4fv(uniforms.proj, false, perspective(0.8, w / h, radius * 0.02, radius * 60));
                    for (const item of model.items) {
                        gl.uniformMatrix4fv(uniforms.model, false, item.model);
                        gl.uniform4fv(uniforms.base, item.base);
                        gl.uniform1f(uniforms.hasTex, item.texture ? 1 : 0);
                        gl.uniform1f(uniforms.hasNormals, item.normal ? 1 : 0);
                        if (item.texture) { gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, item.texture); gl.uniform1i(uniforms.tex, 0); }
                        const attrib = (location, a) => {
                            if (location < 0) return;
                            if (!a) { gl.disableVertexAttribArray(location); gl.vertexAttrib3f(location, 0, 0, 1); return; }
                            gl.bindBuffer(gl.ARRAY_BUFFER, a.buffer); gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, a.size, gl.FLOAT, false, 0, 0);
                        };
                        attrib(loc.pos, item.pos); attrib(loc.normal, item.normal); attrib(loc.uv, item.uv);
                        if (item.doubleSided) gl.disable(gl.CULL_FACE); else gl.disable(gl.CULL_FACE);
                        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, item.indices);
                        gl.drawElements(gl.TRIANGLES, item.count, item.indexType, 0);
                    }
                    raf = requestAnimationFrame(draw);
                };
                draw();
                // Lets a test or the page see that the model is up and what it measures.
                root.dataset.ready = "1";
                root.dataset.meshes = String(model.items.length);
                root.dataset.triangles = String(model.items.reduce((a, i) => a + i.count / 3, 0));
            } catch (e) {
                status.textContent = "Could not show this model: " + (e && e.message ? e.message : e);
                status.classList.add("error");
            }
        });
    };
})();
