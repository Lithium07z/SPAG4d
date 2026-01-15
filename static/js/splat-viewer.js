// Gaussian Splat Viewer
// Simplified viewer for SPLAT format preview

class SplatViewer {
    constructor(container) {
        this.container = container;
        this.canvas = null;
        this.gl = null;
        this.data = null;

        // Camera state (WASD Fly Camera)
        this.position = { x: 0, y: 0, z: 0 };
        this.rotation = { yaw: 0, pitch: 0 };
        this.keys = { w: false, a: false, s: false, d: false, q: false, e: false };
        this.speed = 0.05;
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        this.init();
    }

    init() {
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.container.appendChild(this.canvas);

        // Get WebGL context
        this.gl = this.canvas.getContext('webgl2') || this.canvas.getContext('webgl');

        if (!this.gl) {
            console.error('WebGL not supported');
            this.showMessage('WebGL not supported');
            return;
        }

        // Set up basic shaders
        this.setupShaders();

        // Event listeners
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        document.addEventListener('mousemove', (e) => this.onMouseMove(e));
        document.addEventListener('mouseup', () => this.onMouseUp());
        this.canvas.addEventListener('wheel', (e) => this.onWheel(e));

        // Keyboard controls
        window.addEventListener('keydown', (e) => this.onKeyDown(e));
        window.addEventListener('keyup', (e) => this.onKeyUp(e));

        window.addEventListener('resize', () => this.onResize());

        this.onResize();
    }

    setupShaders() {
        const gl = this.gl;

        // Vertex shader for point rendering
        const vsSource = `
            attribute vec3 aPosition;
            attribute vec3 aColor;
            attribute float aOpacity;
            
            uniform mat4 uModelViewMatrix;
            uniform mat4 uProjectionMatrix;
            uniform float uPointSize;
            
            varying vec3 vColor;
            varying float vOpacity;
            
            void main() {
                gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aPosition, 1.0);
                gl_PointSize = uPointSize / gl_Position.w;
                vColor = aColor;
                vOpacity = aOpacity;
            }
        `;

        // Fragment shader
        const fsSource = `
            precision mediump float;
            
            varying vec3 vColor;
            varying float vOpacity;
            
            void main() {
                float dist = length(gl_PointCoord - vec2(0.5));
                if (dist > 0.5) discard;
                
                float alpha = smoothstep(0.5, 0.3, dist) * vOpacity;
                gl_FragColor = vec4(vColor, alpha);
            }
        `;

        // Compile shaders
        const vertexShader = this.compileShader(gl.VERTEX_SHADER, vsSource);
        const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fsSource);

        if (!vertexShader || !fragmentShader) return;

        // Create program
        this.program = gl.createProgram();
        gl.attachShader(this.program, vertexShader);
        gl.attachShader(this.program, fragmentShader);
        gl.linkProgram(this.program);

        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
            console.error('Shader program failed:', gl.getProgramInfoLog(this.program));
            return;
        }

        // Get locations
        this.attribLocations = {
            position: gl.getAttribLocation(this.program, 'aPosition'),
            color: gl.getAttribLocation(this.program, 'aColor'),
            opacity: gl.getAttribLocation(this.program, 'aOpacity'),
        };

        this.uniformLocations = {
            modelViewMatrix: gl.getUniformLocation(this.program, 'uModelViewMatrix'),
            projectionMatrix: gl.getUniformLocation(this.program, 'uProjectionMatrix'),
            pointSize: gl.getUniformLocation(this.program, 'uPointSize'),
        };
    }

    compileShader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }

        return shader;
    }

    async loadSplat(url) {
        console.log('[SplatViewer] Loading SPLAT from:', url);
        this.showMessage('Loading...');

        try {
            const response = await fetch(url);
            console.log('[SplatViewer] Fetch response:', response.status, response.statusText);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const buffer = await response.arrayBuffer();
            console.log('[SplatViewer] Buffer size:', buffer.byteLength, 'bytes');

            this.parseSplat(buffer);
            this.createBuffers();

            // Hide placeholder
            const placeholder = this.container.querySelector('.viewer-placeholder');
            if (placeholder) {
                placeholder.style.display = 'none';
            }

            // Hide loading message
            const msg = this.container.querySelector('.viewer-message');
            if (msg) msg.style.display = 'none';

            console.log('[SplatViewer] Starting render loop');
            this.render();

        } catch (error) {
            console.error('[SplatViewer] Failed to load SPLAT:', error);
            this.showMessage('Failed to load: ' + error.message);
        }
    }

    parseSplat(buffer) {
        const view = new DataView(buffer);

        // Check magic
        const magic = String.fromCharCode(
            view.getUint8(0), view.getUint8(1), view.getUint8(2),
            view.getUint8(3), view.getUint8(4)
        );

        if (magic !== 'SPLAT') {
            throw new Error('Invalid SPLAT file');
        }

        const vertexCount = view.getUint32(5, true);
        console.log(`Loading ${vertexCount} splats`);

        // Parse vertices
        const positions = new Float32Array(vertexCount * 3);
        const colors = new Float32Array(vertexCount * 3);
        const opacities = new Float32Array(vertexCount);

        let offset = 9;  // After header

        for (let i = 0; i < vertexCount; i++) {
            // Position (3 x float16 = 6 bytes)
            positions[i * 3 + 0] = this.float16ToFloat32(view.getUint16(offset, true));
            positions[i * 3 + 1] = this.float16ToFloat32(view.getUint16(offset + 2, true));
            positions[i * 3 + 2] = this.float16ToFloat32(view.getUint16(offset + 4, true));
            offset += 6;

            // Skip scale (3 bytes)
            offset += 3;

            // Color (3 x uint8)
            colors[i * 3 + 0] = view.getUint8(offset) / 255;
            colors[i * 3 + 1] = view.getUint8(offset + 1) / 255;
            colors[i * 3 + 2] = view.getUint8(offset + 2) / 255;
            offset += 3;

            // Opacity (1 x uint8)
            opacities[i] = view.getUint8(offset) / 255;
            offset += 1;

            // Skip quaternion (4 bytes) and padding (15 bytes)
            offset += 4 + 15;
        }

        this.data = { positions, colors, opacities, count: vertexCount };
    }

    float16ToFloat32(h) {
        const s = (h & 0x8000) >> 15;
        const e = (h & 0x7C00) >> 10;
        const f = h & 0x03FF;

        if (e === 0) {
            return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
        }
        if (e === 31) {
            return f ? NaN : ((s ? -1 : 1) * Infinity);
        }

        return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }

    createBuffers() {
        const gl = this.gl;

        // Position buffer
        this.positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.data.positions, gl.STATIC_DRAW);

        // Color buffer
        this.colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.data.colors, gl.STATIC_DRAW);

        // Opacity buffer
        this.opacityBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.opacityBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.data.opacities, gl.STATIC_DRAW);
    }

    render() {
        if (!this.data || !this.gl) return;

        const gl = this.gl;

        // Clear
        gl.clearColor(0.07, 0.07, 0.1, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        gl.useProgram(this.program);

        // Update camera (WASD)
        this.updateCamera();

        // Model-view matrix (fly camera)
        const modelView = this.createViewMatrix();
        gl.uniformMatrix4fv(this.uniformLocations.modelViewMatrix, false, modelView);

        // Projection matrix
        const aspect = this.canvas.width / this.canvas.height;
        const projection = this.createProjectionMatrix(60, aspect, 0.1, 1000);
        gl.uniformMatrix4fv(this.uniformLocations.projectionMatrix, false, projection);

        // Point size
        gl.uniform1f(this.uniformLocations.pointSize, 20.0);

        // Bind position
        gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
        gl.vertexAttribPointer(this.attribLocations.position, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.attribLocations.position);

        // Bind color
        gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
        gl.vertexAttribPointer(this.attribLocations.color, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.attribLocations.color);

        // Bind opacity
        gl.bindBuffer(gl.ARRAY_BUFFER, this.opacityBuffer);
        gl.vertexAttribPointer(this.attribLocations.opacity, 1, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.attribLocations.opacity);

        // Draw
        gl.drawArrays(gl.POINTS, 0, this.data.count);

        requestAnimationFrame(() => this.render());
    }

    updateCamera() {
        if (!this.keys) return;

        const speed = this.speed * (this.keys.shift ? 2.0 : 1.0);
        const yaw = this.rotation.yaw;

        // Forward/Backward (Z) - fly direction
        // In our coordinate system, Z is backward, so -Z is forward?
        // Let's assume standard OpenGL: -Z is forward.
        // We move in direction of yaw.

        const sinYaw = Math.sin(yaw);
        const cosYaw = Math.cos(yaw);

        if (this.keys.w) {
            this.position.x -= sinYaw * speed;
            this.position.z -= cosYaw * speed;
        }
        if (this.keys.s) {
            this.position.x += sinYaw * speed;
            this.position.z += cosYaw * speed;
        }

        // Left/Right (X)
        if (this.keys.a) {
            this.position.x -= cosYaw * speed;
            this.position.z += sinYaw * speed;
        }
        if (this.keys.d) {
            this.position.x += cosYaw * speed;
            this.position.z -= sinYaw * speed;
        }

        // Up/Down (Y)
        if (this.keys.e || this.keys.space) {
            this.position.y += speed;
        }
        if (this.keys.q || this.keys.ctrl) {
            this.position.y -= speed;
        }
    }

    createViewMatrix() {
        const pitch = this.rotation.pitch;
        const yaw = this.rotation.yaw;

        const cosPitch = Math.cos(pitch);
        const sinPitch = Math.sin(pitch);
        const cosYaw = Math.cos(yaw);
        const sinYaw = Math.sin(yaw);

        // Forward vector
        // In coordinate system where Y is up:
        // x = sin(yaw) * cos(pitch)
        // y = sin(pitch)
        // z = cos(yaw) * cos(pitch)
        const forward = [
            sinYaw * cosPitch,
            sinPitch,
            cosYaw * cosPitch
        ];

        // Target: eye - forward (looking into -Z)
        const target = [
            this.position.x - forward[0],
            this.position.y - forward[1],
            this.position.z - forward[2]
        ];

        return this.lookAt(
            [this.position.x, this.position.y, this.position.z],
            target,
            [0, 1, 0]
        );
    }

    resetView() {
        this.position = { x: 0, y: 0, z: 0 };
        this.rotation = { yaw: 0, pitch: 0 };
    }

    lookAt(eye, target, up) {
        const zAxis = this.normalize([
            eye[0] - target[0],
            eye[1] - target[1],
            eye[2] - target[2]
        ]);
        const xAxis = this.normalize(this.cross(up, zAxis));
        const yAxis = this.cross(zAxis, xAxis);

        return new Float32Array([
            xAxis[0], yAxis[0], zAxis[0], 0,
            xAxis[1], yAxis[1], zAxis[1], 0,
            xAxis[2], yAxis[2], zAxis[2], 0,
            -this.dot(xAxis, eye), -this.dot(yAxis, eye), -this.dot(zAxis, eye), 1
        ]);
    }

    createProjectionMatrix(fov, aspect, near, far) {
        const f = 1 / Math.tan(fov * Math.PI / 360);
        const nf = 1 / (near - far);

        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, 2 * far * near * nf, 0
        ]);
    }

    normalize(v) {
        const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        return [v[0] / len, v[1] / len, v[2] / len];
    }

    cross(a, b) {
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ];
    }

    dot(a, b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    onMouseDown(e) {
        // Capture pointer for smoother look
        if (this.canvas.requestPointerLock) {
            this.canvas.requestPointerLock();
        }

        this.isDragging = true;
    }

    onMouseMove(e) {
        // Support both pointer lock and drag
        if (document.pointerLockElement === this.canvas) {
            this.rotation.yaw -= e.movementX * 0.002;
            this.rotation.pitch -= e.movementY * 0.002;
        } else if (this.isDragging) {
            this.rotation.yaw -= e.movementX * 0.005;
            this.rotation.pitch -= e.movementY * 0.005;
        } else {
            return;
        }

        // Clamp pitch to avoid gimbal lock flip
        const maxPitch = Math.PI / 2 - 0.01;
        this.rotation.pitch = Math.max(-maxPitch, Math.min(maxPitch, this.rotation.pitch));
    }

    onMouseUp() {
        this.isDragging = false;
        if (document.exitPointerLock) {
            document.exitPointerLock();
        }
    }

    onKeyDown(e) {
        const key = e.key.toLowerCase();
        if (this.keys.hasOwnProperty(key)) this.keys[key] = true;
        if (e.key === 'Shift') this.keys.shift = true;
        if (e.key === 'Control') this.keys.ctrl = true;
        if (e.code === 'Space') this.keys.space = true;

        if (key === 'r') this.resetView(); // Hotkey reset
    }

    onKeyUp(e) {
        const key = e.key.toLowerCase();
        if (this.keys.hasOwnProperty(key)) this.keys[key] = false;
        if (e.key === 'Shift') this.keys.shift = false;
        if (e.key === 'Control') this.keys.ctrl = false;
        if (e.code === 'Space') this.keys.space = false;
    }

    onWheel(e) {
        e.preventDefault();
        // Adjust speed with scroll
        this.speed *= (e.deltaY > 0 ? 0.9 : 1.1);
        this.speed = Math.max(0.01, Math.min(5.0, this.speed));
    }

    onResize() {
        if (!this.canvas) return;

        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.canvas.width = width * window.devicePixelRatio;
        this.canvas.height = height * window.devicePixelRatio;

        if (this.gl) {
            this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    showMessage(text) {
        // Add temporary message overlay
        let msg = this.container.querySelector('.viewer-message');
        if (!msg) {
            msg = document.createElement('div');
            msg.className = 'viewer-message';
            msg.style.cssText = `
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: #64748b;
                font-size: 0.9rem;
            `;
            this.container.appendChild(msg);
        }
        msg.textContent = text;
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SplatViewer;
}
