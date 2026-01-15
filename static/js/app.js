// SPAG-4D Main Application JS

class SPAG4DApp {
    constructor() {
        this.currentFile = null;
        this.currentJobId = null;
        this.pollInterval = null;

        this.panoViewer = null;
        this.splatViewer = null;

        this.init();
    }

    init() {
        // DOM Elements
        this.fileInput = document.getElementById('file-input');
        this.fileLabel = document.getElementById('filename');
        this.convertBtn = document.getElementById('convert-btn');
        this.downloadPlyBtn = document.getElementById('download-ply-btn');
        this.downloadSplatBtn = document.getElementById('download-splat-btn');
        this.statusText = document.getElementById('status-text');
        this.progressText = document.getElementById('progress-text');
        this.gpuStatus = document.getElementById('gpu-status');

        // Parameters
        this.strideSelect = document.getElementById('stride');
        this.scaleFactorInput = document.getElementById('scale-factor');
        this.thicknessInput = document.getElementById('thickness');
        this.globalScaleInput = document.getElementById('global-scale');
        this.depthMinInput = document.getElementById('depth-min');
        this.depthMaxInput = document.getElementById('depth-max');

        // Event Listeners
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        this.convertBtn.addEventListener('click', () => this.startConversion());
        this.downloadPlyBtn.addEventListener('click', () => this.downloadFile('ply'));
        this.downloadSplatBtn.addEventListener('click', () => this.downloadFile('splat'));

        // Reset View Button
        const resetBtn = document.getElementById('reset-view-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                if (this.splatViewer) this.splatViewer.resetView();
            });
        }

        // Initialize viewers
        this.initViewers();

        // Check health
        this.checkHealth();
        setInterval(() => this.checkHealth(), 30000);

        // Preload test image if available
        this.preloadTestImage();
    }

    async preloadTestImage() {
        const testImagePath = '/TestImage/monbachtal_riverbank_primary.jpg';
        try {
            const response = await fetch(testImagePath, { method: 'HEAD' });
            if (response.ok) {
                // Load test image into pano viewer
                if (this.panoViewer) {
                    console.log('[App] Preloading test image');
                    this.panoViewer.loadImage(testImagePath);
                }

                // Fetch and set as current file for conversion
                const imgResponse = await fetch(testImagePath);
                const blob = await imgResponse.blob();
                const file = new File([blob], 'monbachtal_riverbank_primary.jpg', { type: 'image/jpeg' });

                this.currentFile = file;
                this.fileLabel.textContent = 'monbachtal_riverbank_primary.jpg (demo)';
                this.convertBtn.disabled = false;
                this.setStatus('Demo image loaded - ready to convert');
            }
        } catch (e) {
            console.log('[App] No test image available for preload');
        }
    }

    initViewers() {
        const panoContainer = document.getElementById('pano-viewer');
        const splatContainer = document.getElementById('splat-viewer');

        // Initialize 360 viewer
        if (typeof PanoViewer !== 'undefined') {
            this.panoViewer = new PanoViewer(panoContainer);
        }

        // Initialize splat viewer
        if (typeof SplatViewer !== 'undefined') {
            this.splatViewer = new SplatViewer(splatContainer);
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        console.log('[App] File selected:', file ? file.name : 'none');
        if (!file) return;

        this.currentFile = file;
        this.fileLabel.textContent = file.name;
        this.convertBtn.disabled = false;
        this.downloadPlyBtn.disabled = true;
        this.downloadSplatBtn.disabled = true;

        // Load into 360 viewer
        if (this.panoViewer) {
            const url = URL.createObjectURL(file);
            console.log('[App] Loading pano from blob URL:', url);
            this.panoViewer.loadImage(url);
        } else {
            console.warn('[App] PanoViewer not initialized');
        }

        this.setStatus('Ready to convert');
    }

    async startConversion() {
        if (!this.currentFile) return;

        this.convertBtn.disabled = true;
        this.setStatus('Uploading...', 'Preparing');

        // Prepare form data
        const formData = new FormData();
        formData.append('file', this.currentFile);

        // Add parameters as query string
        const params = new URLSearchParams({
            stride: this.strideSelect.value,
            scale_factor: this.scaleFactorInput.value,
            thickness: this.thicknessInput.value,
            global_scale: this.globalScaleInput.value,
            depth_min: this.depthMinInput.value,
            depth_max: this.depthMaxInput.value
        });

        try {
            const response = await fetch(`/api/convert?${params}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }

            const result = await response.json();
            this.currentJobId = result.job_id;

            this.setStatus('Processing...', `Queue position: ${result.queue_position}`);
            this.startPolling();

        } catch (error) {
            this.setStatus(`Error: ${error.message}`);
            this.convertBtn.disabled = false;
        }
    }

    startPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }

        this.pollInterval = setInterval(() => this.checkJobStatus(), 1000);
    }

    async checkJobStatus() {
        if (!this.currentJobId) return;

        try {
            const response = await fetch(`/api/status/${this.currentJobId}`);

            // Handle non-OK responses
            if (!response.ok) {
                throw new Error(`Status check failed: ${response.status}`);
            }

            const status = await response.json();

            // Reset error count on success
            this.pollErrorCount = 0;

            if (status.status === 'queued') {
                this.setStatus('Waiting...', `Queue position: ${status.queue_position}`);

            } else if (status.status === 'processing') {
                this.setStatus('Processing...', 'GPU active');

            } else if (status.status === 'complete') {
                clearInterval(this.pollInterval);
                this.pollInterval = null;

                this.setStatus('Complete!',
                    `${status.splat_count.toLocaleString()} splats • ${status.file_size_mb} MB • ${status.processing_time}s`
                );

                this.convertBtn.disabled = false;
                this.downloadPlyBtn.disabled = false;
                this.downloadSplatBtn.disabled = false;

                // Load preview into splat viewer
                if (this.splatViewer && status.preview_url) {
                    this.splatViewer.loadSplat(status.preview_url);
                }

            } else if (status.status === 'error') {
                clearInterval(this.pollInterval);
                this.pollInterval = null;

                this.setStatus(`Error: ${status.error}`);
                this.convertBtn.disabled = false;
            }

        } catch (error) {
            // Handle polling errors - stop polling and show error after multiple failures
            console.error('Polling error:', error);
            this.pollErrorCount = (this.pollErrorCount || 0) + 1;

            if (this.pollErrorCount >= 5) {
                clearInterval(this.pollInterval);
                this.pollInterval = null;
                this.setStatus('Connection error - please try again');
                this.convertBtn.disabled = false;
                this.pollErrorCount = 0;
            }
        }
    }

    downloadFile(format) {
        if (!this.currentJobId) return;

        const url = `/api/download/${this.currentJobId}?format=${format}`;
        const link = document.createElement('a');
        link.href = url;
        link.download = `spag4d_${this.currentJobId.slice(0, 8)}.${format}`;
        link.click();
    }

    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const health = await response.json();

            if (health.gpu_available) {
                this.gpuStatus.className = 'gpu-status';
            } else if (health.active_jobs > 0) {
                this.gpuStatus.className = 'gpu-status busy';
            } else {
                this.gpuStatus.className = 'gpu-status error';
            }

        } catch (error) {
            this.gpuStatus.className = 'gpu-status error';
        }
    }

    setStatus(text, progress = '') {
        this.statusText.textContent = text;
        this.progressText.textContent = progress;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SPAG4DApp();
});
