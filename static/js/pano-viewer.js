// 360° Panorama Viewer using Three.js

class PanoViewer {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.sphere = null;
        this.isUserInteracting = false;
        this.lon = 0;
        this.lat = 0;
        this.onPointerDownLon = 0;
        this.onPointerDownLat = 0;
        this.onPointerDownX = 0;
        this.onPointerDownY = 0;

        this.projection = 'sphere'; // 'sphere' or 'flat'
        this.texture = null;
        this.zoom = 1.0;
        this.panX = 0;
        this.panY = 0;

        this.init();
    }

    init() {
        // Remove placeholder
        const placeholder = this.container.querySelector('.viewer-placeholder');

        // Scene
        this.scene = new THREE.Scene();

        // Camera
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1100);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        // Initial geometry (Sphere)
        this.createGeometry();

        // Event listeners
        this.container.addEventListener('pointerdown', (e) => this.onPointerDown(e));
        this.container.addEventListener('pointermove', (e) => this.onPointerMove(e));
        this.container.addEventListener('pointerup', () => this.onPointerUp());
        this.container.addEventListener('wheel', (e) => this.onWheel(e));

        // Use ResizeObserver for robust layout handling
        this.resizeObserver = new ResizeObserver(() => this.onResize());
        this.resizeObserver.observe(this.container);

        // Start render loop
        this.animate();
    }

    createGeometry() {
        // Clean up existing
        if (this.mesh) {
            this.scene.remove(this.mesh);
            this.mesh.geometry.dispose();
            // Reuse material if possible, but recreating is safer for mapping changes
        }

        if (this.projection === 'sphere') {
            const geometry = new THREE.SphereGeometry(500, 60, 40);
            geometry.scale(-1, 1, 1); // Invert

            this.camera.position.set(0, 0, 0);

            const material = new THREE.MeshBasicMaterial({
                map: this.texture,
                color: this.texture ? 0xffffff : 0x333344
            });

            this.mesh = new THREE.Mesh(geometry, material);
            this.scene.add(this.mesh);

        } else {
            // Flat plane
            // Aspect ratio will be set when texture loads or default 2:1
            const aspect = this.texture ? (this.texture.image.width / this.texture.image.height) : 2.0;
            const height = 500;
            const width = height * aspect;

            const geometry = new THREE.PlaneGeometry(width, height);

            // Camera needs to step back to see it
            // FOV 75, height 500... 
            // tan(fov/2) = (h/2) / dist => dist = (h/2) / tan(fov/2)
            const dist = (height / 2) / Math.tan(THREE.MathUtils.degToRad(75 / 2));
            this.camera.position.set(0, 0, dist);
            this.camera.lookAt(0, 0, 0);

            const material = new THREE.MeshBasicMaterial({
                map: this.texture,
                color: this.texture ? 0xffffff : 0x333344,
                side: THREE.DoubleSide
            });

            this.mesh = new THREE.Mesh(geometry, material);
            this.scene.add(this.mesh);
        }
    }

    loadImage(url) {
        console.log('[PanoViewer] Loading image:', url);
        const loader = new THREE.TextureLoader();

        loader.load(
            url,
            (texture) => {
                console.log('[PanoViewer] Texture loaded successfully');
                texture.colorSpace = THREE.SRGBColorSpace;
                this.texture = texture;

                // Recreate with new texture
                this.createGeometry();

                // Remove placeholder if still present
                const placeholder = this.container.querySelector('.viewer-placeholder');
                if (placeholder) {
                    placeholder.style.display = 'none';
                }
            },
            (progress) => {
                console.log('[PanoViewer] Loading progress:', progress);
            },
            (error) => {
                console.error('[PanoViewer] Error loading texture:', error);
            }
        );
    }

    setProjection(mode) {
        if (this.projection === mode) return;
        this.projection = mode;
        this.createGeometry();

        // Reset view params
        this.lon = 0;
        this.lat = 0;
        this.panX = 0;
        this.panY = 0;
        this.zoom = 1.0;
        this.camera.fov = 75;
        this.camera.updateProjectionMatrix();
    }

    onPointerDown(event) {
        this.isUserInteracting = true;
        this.onPointerDownX = event.clientX;
        this.onPointerDownY = event.clientY;
        this.onPointerDownLon = this.lon;
        this.onPointerDownLat = this.lat;
        this.onPointerDownPanX = this.panX;
        this.onPointerDownPanY = this.panY;
    }

    onPointerMove(event) {
        if (!this.isUserInteracting) return;

        if (this.projection === 'sphere') {
            this.lon = (event.clientX - this.onPointerDownX) * 0.1 + this.onPointerDownLon;
            this.lat = (event.clientY - this.onPointerDownY) * 0.1 + this.onPointerDownLat;
            this.lat = Math.max(-85, Math.min(85, this.lat));
        } else {
            // Flat pan
            const scale = 0.5 * this.zoom; // Adjust sensitivity
            this.panX = (event.clientX - this.onPointerDownX) * scale + this.onPointerDownPanX;
            this.panY = (this.onPointerDownY - event.clientY) * scale + this.onPointerDownPanY; // Invert Y

            this.camera.position.x = -this.panX;
            this.camera.position.y = -this.panY;
        }
    }

    onPointerUp() {
        this.isUserInteracting = false;
    }

    onWheel(event) {
        event.preventDefault();

        if (this.projection === 'sphere') {
            const fov = this.camera.fov + event.deltaY * 0.05;
            this.camera.fov = Math.max(30, Math.min(100, fov));
            this.camera.updateProjectionMatrix();
        } else {
            // Flat zoom (move camera Z)
            const speed = event.deltaY * 0.5;
            this.camera.position.z += speed;
            this.camera.position.z = Math.max(100, Math.min(2000, this.camera.position.z));
            this.zoom = this.camera.position.z / 500; // Approx logic
        }
    }

    onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.update();
    }

    update() {
        if (this.projection === 'sphere') {
            // Convert spherical to Cartesian
            const phi = THREE.MathUtils.degToRad(90 - this.lat);
            const theta = THREE.MathUtils.degToRad(this.lon);

            const target = new THREE.Vector3(
                Math.sin(phi) * Math.cos(theta),
                Math.cos(phi),
                Math.sin(phi) * Math.sin(theta)
            );

            this.camera.lookAt(target);
        }
        // Flat mode: camera position is updated in events, no lookAt update needed per frame
        // unless we want momentum. For now, static is fine.

        this.renderer.render(this.scene, this.camera);
    }

    dispose() {
        // ... (rest is fine)
        if (this.renderer) {
            this.renderer.dispose();
        }
        if (this.mesh) {
            this.mesh.geometry.dispose();
            this.mesh.material.dispose();
        }
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PanoViewer;
}
