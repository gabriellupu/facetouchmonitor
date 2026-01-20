/**
 * Face Touch Monitor - MediaPipe-powered face touch detection
 * Uses MediaPipe Face Mesh and Hand Landmarker for accurate proximity detection
 */

import { FaceLandmarker, HandLandmarker, FilesetResolver, DrawingUtils } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs';

// ============================================================================
// State Management
// ============================================================================

const state = {
    isRunning: false,
    faceLandmarker: null,
    handLandmarker: null,
    video: null,
    canvas: null,
    ctx: null,
    drawingUtils: null,
    animationId: null,

    // Statistics
    touchCount: 0,
    lastTouchTime: null,
    startTime: null,
    lastFrameTime: 0,
    fps: 0,

    // Alert state
    alertCooldown: false,
    isTouching: false,
    wasTouching: false,

    // Audio context (lazy init)
    audioContext: null,

    // Settings
    settings: {
        sensitivity: 100,
        alertCooldownMs: 3000,
        soundEnabled: true,
        notifyEnabled: false,
        visualAlertEnabled: true,
        showFaceMesh: true,
        showHands: true,
        showProximity: false,
        // Detection zones - which regions trigger alerts
        zones: {
            mouth: true,
            nose: true,
            leftEye: true,
            rightEye: true,
            leftCheek: false,
            rightCheek: false,
            chin: false
        }
    },

    // Last touched zone (for status display)
    lastTouchedZone: null
};

// Face regions for nail-biting / face touch detection
// These indices correspond to MediaPipe Face Mesh landmarks
const FACE_REGIONS = {
    // Mouth region - critical for nail biting detection
    mouth: [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415],
    // Nose region
    nose: [1, 2, 4, 5, 6, 19, 94, 168, 195, 197, 236, 237, 238, 239, 240, 241, 242, 250, 456, 457, 458, 459, 460, 461, 462],
    // Left eye region
    leftEye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    // Right eye region
    rightEye: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    // Cheeks
    leftCheek: [117, 118, 119, 120, 121, 126, 142, 203, 206, 216, 207, 187],
    rightCheek: [346, 347, 348, 349, 350, 355, 371, 423, 426, 436, 427, 411],
    // Chin
    chin: [152, 175, 176, 148, 149, 150, 136, 169, 170, 171, 377, 378, 379, 365, 397, 288, 361, 323]
};

// Hand fingertip indices (MediaPipe Hand Landmarks)
const FINGERTIPS = [4, 8, 12, 16, 20]; // thumb, index, middle, ring, pinky
const FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky'];

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
    video: document.getElementById('videoElement'),
    canvas: document.getElementById('overlayCanvas'),
    welcomeContent: document.getElementById('welcomeContent'),
    loadingState: document.getElementById('loadingState'),
    startButton: document.getElementById('startButton'),
    stopButton: document.getElementById('stopButton'),
    resetStats: document.getElementById('resetStats'),

    // Status elements
    detectionStatus: document.getElementById('detectionStatus'),
    connectionStatus: document.getElementById('connectionStatus'),
    statusText: document.getElementById('statusText'),
    alertFlash: document.getElementById('alertFlash'),

    // Stats elements
    touchCount: document.getElementById('touchCount'),
    lastTouch: document.getElementById('lastTouch'),
    touchRate: document.getElementById('touchRate'),
    fpsDisplay: document.getElementById('fpsDisplay'),

    // Control elements
    beepToggle: document.getElementById('beepToggle'),
    notifyToggle: document.getElementById('notifyToggle'),
    visualAlertToggle: document.getElementById('visualAlertToggle'),
    alertCooldown: document.getElementById('alertCooldown'),
    showLandmarks: document.getElementById('showLandmarks'),
    showHands: document.getElementById('showHands'),
    showProximity: document.getElementById('showProximity'),
    sensitivitySlider: document.getElementById('sensitivitySlider'),
    sensitivityValue: document.getElementById('sensitivityValue'),

    // Zone toggles
    zoneMouth: document.getElementById('zoneMouth'),
    zoneNose: document.getElementById('zoneNose'),
    zoneEyes: document.getElementById('zoneEyes'),
    zoneCheeks: document.getElementById('zoneCheeks'),
    zoneChin: document.getElementById('zoneChin')
};

// ============================================================================
// Initialization
// ============================================================================

async function initializeMediaPipe() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
        );

        // Initialize Face Landmarker
        state.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numFaces: 1,
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false
        });

        // Initialize Hand Landmarker
        state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numHands: 2,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        return true;
    } catch (error) {
        console.error('Failed to initialize MediaPipe:', error);
        return false;
    }
}

async function initializeCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            },
            audio: false
        });

        elements.video.srcObject = stream;

        return new Promise((resolve) => {
            elements.video.onloadedmetadata = () => {
                elements.video.play();

                // Set canvas dimensions to match video
                elements.canvas.width = elements.video.videoWidth;
                elements.canvas.height = elements.video.videoHeight;

                state.video = elements.video;
                state.canvas = elements.canvas;
                state.ctx = elements.canvas.getContext('2d');
                state.drawingUtils = new DrawingUtils(state.ctx);

                resolve(true);
            };
        });
    } catch (error) {
        console.error('Failed to access camera:', error);
        alert('Unable to access camera. Please ensure you have granted camera permissions.');
        return false;
    }
}

// ============================================================================
// Detection Loop
// ============================================================================

function detectFrame() {
    if (!state.isRunning) return;

    const now = performance.now();

    // Calculate FPS
    if (state.lastFrameTime > 0) {
        state.fps = 1000 / (now - state.lastFrameTime);
    }
    state.lastFrameTime = now;

    // Get face and hand landmarks
    const faceResults = state.faceLandmarker.detectForVideo(state.video, now);
    const handResults = state.handLandmarker.detectForVideo(state.video, now);

    // Clear canvas
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);

    // Mirror the canvas to match video
    state.ctx.save();
    state.ctx.scale(-1, 1);
    state.ctx.translate(-state.canvas.width, 0);

    // Process and draw results
    const faceLandmarks = faceResults.faceLandmarks?.[0] || null;
    const handLandmarksList = handResults.landmarks || [];

    // Draw visualizations
    if (faceLandmarks) {
        if (state.settings.showFaceMesh) {
            drawFaceMesh(faceLandmarks);
        }
        if (state.settings.showProximity) {
            drawProximityZones(faceLandmarks);
        }
    }

    if (handLandmarksList.length > 0 && state.settings.showHands) {
        drawHands(handLandmarksList);
    }

    // Check for face touch
    const touchDetected = checkFaceTouch(faceLandmarks, handLandmarksList);

    // Handle touch state changes
    handleTouchState(touchDetected, faceLandmarks !== null);

    state.ctx.restore();

    // Update UI
    updateUI();

    // Schedule next frame
    state.animationId = requestAnimationFrame(detectFrame);
}

// ============================================================================
// Face Touch Detection
// ============================================================================

function checkFaceTouch(faceLandmarks, handLandmarksList) {
    if (!faceLandmarks || handLandmarksList.length === 0) {
        state.lastTouchedZone = null;
        return false;
    }

    const videoWidth = state.canvas.width;
    const videoHeight = state.canvas.height;

    // Determine which zones are enabled based on settings
    const enabledZones = [];
    if (state.settings.zones.mouth) enabledZones.push('mouth');
    if (state.settings.zones.nose) enabledZones.push('nose');
    if (state.settings.zones.leftEye) enabledZones.push('leftEye');
    if (state.settings.zones.rightEye) enabledZones.push('rightEye');
    if (state.settings.zones.leftCheek) enabledZones.push('leftCheek');
    if (state.settings.zones.rightCheek) enabledZones.push('rightCheek');
    if (state.settings.zones.chin) enabledZones.push('chin');

    // If no zones enabled, no detection
    if (enabledZones.length === 0) {
        state.lastTouchedZone = null;
        return false;
    }

    // Convert face landmarks to pixel coordinates for enabled zones only
    const facePoints = {};
    for (const region of enabledZones) {
        const indices = FACE_REGIONS[region];
        if (indices) {
            facePoints[region] = indices.map(idx => {
                const lm = faceLandmarks[idx];
                return {
                    x: lm.x * videoWidth,
                    y: lm.y * videoHeight,
                    z: lm.z * videoWidth
                };
            });
        }
    }

    // Check each hand
    for (const handLandmarks of handLandmarksList) {
        // Check fingertips
        for (let i = 0; i < FINGERTIPS.length; i++) {
            const tipIdx = FINGERTIPS[i];
            const tip = handLandmarks[tipIdx];
            const tipPoint = {
                x: tip.x * videoWidth,
                y: tip.y * videoHeight,
                z: tip.z * videoWidth
            };

            // Check against each enabled face region
            for (const [region, points] of Object.entries(facePoints)) {
                const minDist = getMinDistance(tipPoint, points);

                // Base threshold adjusted by sensitivity
                // Lower sensitivity = larger threshold = less sensitive
                const baseThreshold = 40; // pixels
                const sensitivityMultiplier = state.settings.sensitivity / 100;
                const threshold = baseThreshold / sensitivityMultiplier;

                if (minDist < threshold) {
                    // Additional z-depth check to reduce false positives
                    // Hand should be roughly at same depth or closer than face
                    const avgFaceZ = points.reduce((sum, p) => sum + p.z, 0) / points.length;
                    const zDiff = tipPoint.z - avgFaceZ;

                    // If hand is significantly behind face, ignore
                    if (zDiff < 50) { // Allow some tolerance
                        state.lastTouchedZone = region;
                        return true;
                    }
                }
            }
        }
    }

    state.lastTouchedZone = null;
    return false;
}

// Helper to get display name for a zone
function getZoneDisplayName(zone) {
    const names = {
        mouth: 'Mouth',
        nose: 'Nose',
        leftEye: 'Left Eye',
        rightEye: 'Right Eye',
        leftCheek: 'Left Cheek',
        rightCheek: 'Right Cheek',
        chin: 'Chin'
    };
    return names[zone] || zone;
}

function getMinDistance(point, targets) {
    let minDist = Infinity;
    for (const target of targets) {
        const dx = point.x - target.x;
        const dy = point.y - target.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < minDist) {
            minDist = dist;
        }
    }
    return minDist;
}

// ============================================================================
// Touch State Handling
// ============================================================================

function handleTouchState(touchDetected, faceVisible) {
    state.wasTouching = state.isTouching;
    state.isTouching = touchDetected;

    // Update status display
    updateDetectionStatus(faceVisible, touchDetected);

    // Trigger alert on new touch
    if (touchDetected && !state.wasTouching && !state.alertCooldown) {
        triggerAlert();
    }
}

function triggerAlert() {
    state.touchCount++;
    state.lastTouchTime = Date.now();

    const zoneName = state.lastTouchedZone ? getZoneDisplayName(state.lastTouchedZone) : 'Face';
    console.log(`${zoneName} touch detected! Count: ${state.touchCount}`);

    // Sound alert
    if (state.settings.soundEnabled) {
        playBeep(440, 150);
    }

    // Visual alert
    if (state.settings.visualAlertEnabled) {
        elements.alertFlash.classList.add('active');
        setTimeout(() => {
            elements.alertFlash.classList.remove('active');
        }, 400);
    }

    // Browser notification
    if (state.settings.notifyEnabled) {
        sendNotification(`${zoneName} touch detected! Count: ${state.touchCount}`);
    }

    // Set cooldown
    state.alertCooldown = true;
    setTimeout(() => {
        state.alertCooldown = false;
    }, state.settings.alertCooldownMs);
}

// ============================================================================
// Drawing Functions
// ============================================================================

function drawFaceMesh(landmarks) {
    // Draw face mesh connectors
    state.ctx.strokeStyle = 'rgba(14, 165, 233, 0.3)';
    state.ctx.lineWidth = 1;

    // Draw key facial features with different colors
    const faceOvalColor = 'rgba(14, 165, 233, 0.5)';
    const lipsColor = 'rgba(239, 68, 68, 0.5)';
    const eyeColor = 'rgba(34, 197, 94, 0.5)';

    // Face oval
    drawLandmarkConnections(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, faceOvalColor);

    // Lips
    drawLandmarkConnections(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, lipsColor);

    // Eyes
    drawLandmarkConnections(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, eyeColor);
    drawLandmarkConnections(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, eyeColor);

    // Eyebrows
    drawLandmarkConnections(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, eyeColor);
    drawLandmarkConnections(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, eyeColor);
}

function drawLandmarkConnections(landmarks, connections, color) {
    if (!connections) return;

    state.ctx.strokeStyle = color;
    state.ctx.lineWidth = 2;

    for (const connection of connections) {
        const start = landmarks[connection.start];
        const end = landmarks[connection.end];

        state.ctx.beginPath();
        state.ctx.moveTo(start.x * state.canvas.width, start.y * state.canvas.height);
        state.ctx.lineTo(end.x * state.canvas.width, end.y * state.canvas.height);
        state.ctx.stroke();
    }
}

function drawHands(handLandmarksList) {
    for (const landmarks of handLandmarksList) {
        // Draw connections
        state.ctx.strokeStyle = state.isTouching ? 'rgba(239, 68, 68, 0.8)' : 'rgba(34, 197, 94, 0.8)';
        state.ctx.lineWidth = 2;

        // Draw hand skeleton
        const connections = HandLandmarker.HAND_CONNECTIONS;
        for (const connection of connections) {
            const start = landmarks[connection.start];
            const end = landmarks[connection.end];

            state.ctx.beginPath();
            state.ctx.moveTo(start.x * state.canvas.width, start.y * state.canvas.height);
            state.ctx.lineTo(end.x * state.canvas.width, end.y * state.canvas.height);
            state.ctx.stroke();
        }

        // Draw landmarks
        for (let i = 0; i < landmarks.length; i++) {
            const lm = landmarks[i];
            const x = lm.x * state.canvas.width;
            const y = lm.y * state.canvas.height;

            // Highlight fingertips
            const isFingertip = FINGERTIPS.includes(i);
            const radius = isFingertip ? 6 : 3;
            const color = isFingertip
                ? (state.isTouching ? 'rgba(239, 68, 68, 1)' : 'rgba(245, 158, 11, 1)')
                : 'rgba(34, 197, 94, 0.8)';

            state.ctx.beginPath();
            state.ctx.arc(x, y, radius, 0, 2 * Math.PI);
            state.ctx.fillStyle = color;
            state.ctx.fill();
        }
    }
}

function drawProximityZones(faceLandmarks) {
    const videoWidth = state.canvas.width;
    const videoHeight = state.canvas.height;

    // Define all zones with their colors (matching CSS toggle colors)
    const allZones = [
        { region: 'mouth', color: 'rgba(239, 68, 68, 0.2)', enabled: state.settings.zones.mouth },
        { region: 'nose', color: 'rgba(245, 158, 11, 0.2)', enabled: state.settings.zones.nose },
        { region: 'leftEye', color: 'rgba(14, 165, 233, 0.2)', enabled: state.settings.zones.leftEye },
        { region: 'rightEye', color: 'rgba(14, 165, 233, 0.2)', enabled: state.settings.zones.rightEye },
        { region: 'leftCheek', color: 'rgba(139, 92, 246, 0.2)', enabled: state.settings.zones.leftCheek },
        { region: 'rightCheek', color: 'rgba(139, 92, 246, 0.2)', enabled: state.settings.zones.rightCheek },
        { region: 'chin', color: 'rgba(34, 197, 94, 0.2)', enabled: state.settings.zones.chin }
    ];

    const threshold = 40 / (state.settings.sensitivity / 100);

    for (const { region, color, enabled } of allZones) {
        // Only draw enabled zones
        if (!enabled) continue;

        const indices = FACE_REGIONS[region];
        if (!indices) continue;

        const points = indices.map(idx => ({
            x: faceLandmarks[idx].x * videoWidth,
            y: faceLandmarks[idx].y * videoHeight
        }));

        // Draw expanded region
        state.ctx.fillStyle = color;
        state.ctx.beginPath();

        // Find bounding box and expand it
        const xs = points.map(p => p.x);
        const ys = points.map(p => p.y);
        const minX = Math.min(...xs) - threshold;
        const maxX = Math.max(...xs) + threshold;
        const minY = Math.min(...ys) - threshold;
        const maxY = Math.max(...ys) + threshold;

        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const radiusX = (maxX - minX) / 2;
        const radiusY = (maxY - minY) / 2;

        state.ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI);
        state.ctx.fill();
    }
}

// ============================================================================
// UI Updates
// ============================================================================

function updateUI() {
    // Update FPS
    elements.fpsDisplay.textContent = Math.round(state.fps);

    // Update touch count
    elements.touchCount.textContent = state.touchCount;

    // Update last touch time
    if (state.lastTouchTime) {
        const secAgo = Math.round((Date.now() - state.lastTouchTime) / 1000);
        elements.lastTouch.textContent = `${secAgo}s ago`;
    }

    // Update touch rate
    if (state.startTime) {
        const hoursElapsed = (Date.now() - state.startTime) / (1000 * 60 * 60);
        if (hoursElapsed > 0) {
            const rate = state.touchCount / hoursElapsed;
            elements.touchRate.textContent = rate.toFixed(1);
        }
    }
}

function updateDetectionStatus(faceVisible, touching) {
    const statusEl = elements.detectionStatus;
    const textEl = statusEl.querySelector('.status-text');

    statusEl.classList.remove('warning', 'danger');

    if (!faceVisible) {
        textEl.textContent = 'No face detected';
        statusEl.classList.add('warning');
    } else if (touching) {
        const zoneName = state.lastTouchedZone ? getZoneDisplayName(state.lastTouchedZone) : 'Face';
        textEl.textContent = `${zoneName} touch detected!`;
        statusEl.classList.add('danger');
    } else {
        textEl.textContent = 'Monitoring...';
    }
}

// ============================================================================
// Audio
// ============================================================================

function playBeep(frequency, duration) {
    try {
        if (!state.audioContext) {
            state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }

        const oscillator = state.audioContext.createOscillator();
        const gainNode = state.audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(state.audioContext.destination);

        oscillator.frequency.value = frequency;
        oscillator.type = 'sine';

        gainNode.gain.setValueAtTime(0.3, state.audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, state.audioContext.currentTime + duration / 1000);

        oscillator.start();
        oscillator.stop(state.audioContext.currentTime + duration / 1000);
    } catch (error) {
        console.warn('Audio playback failed:', error);
    }
}

// ============================================================================
// Notifications
// ============================================================================

async function requestNotificationPermission() {
    if (!('Notification' in window)) {
        console.warn('Notifications not supported');
        return false;
    }

    if (Notification.permission === 'granted') {
        return true;
    }

    if (Notification.permission !== 'denied') {
        const permission = await Notification.requestPermission();
        return permission === 'granted';
    }

    return false;
}

function sendNotification(message) {
    if (Notification.permission === 'granted') {
        new Notification('Face Touch Monitor', {
            body: message,
            icon: 'favicon.ico',
            silent: true
        });
    }
}

// ============================================================================
// Controls & Event Handlers
// ============================================================================

function setupEventListeners() {
    // Start button
    elements.startButton.addEventListener('click', startMonitoring);

    // Stop button
    elements.stopButton.addEventListener('click', stopMonitoring);

    // Reset stats
    elements.resetStats.addEventListener('click', resetStatistics);

    // Toggle controls
    elements.beepToggle.addEventListener('change', (e) => {
        state.settings.soundEnabled = e.target.checked;
    });

    elements.notifyToggle.addEventListener('change', async (e) => {
        if (e.target.checked) {
            const granted = await requestNotificationPermission();
            if (!granted) {
                e.target.checked = false;
                alert('Notification permission was denied. Please enable it in your browser settings.');
                return;
            }
        }
        state.settings.notifyEnabled = e.target.checked;
    });

    elements.visualAlertToggle.addEventListener('change', (e) => {
        state.settings.visualAlertEnabled = e.target.checked;
    });

    elements.alertCooldown.addEventListener('change', (e) => {
        const value = parseInt(e.target.value, 10);
        if (value >= 1 && value <= 30) {
            state.settings.alertCooldownMs = value * 1000;
        }
    });

    // Visualization controls
    elements.showLandmarks.addEventListener('change', (e) => {
        state.settings.showFaceMesh = e.target.checked;
    });

    elements.showHands.addEventListener('change', (e) => {
        state.settings.showHands = e.target.checked;
    });

    elements.showProximity.addEventListener('change', (e) => {
        state.settings.showProximity = e.target.checked;
    });

    // Sensitivity slider
    elements.sensitivitySlider.addEventListener('input', (e) => {
        state.settings.sensitivity = parseInt(e.target.value, 10);
        elements.sensitivityValue.textContent = `${state.settings.sensitivity}%`;
    });

    // Zone toggles
    elements.zoneMouth.addEventListener('change', (e) => {
        state.settings.zones.mouth = e.target.checked;
    });

    elements.zoneNose.addEventListener('change', (e) => {
        state.settings.zones.nose = e.target.checked;
    });

    elements.zoneEyes.addEventListener('change', (e) => {
        state.settings.zones.leftEye = e.target.checked;
        state.settings.zones.rightEye = e.target.checked;
    });

    elements.zoneCheeks.addEventListener('change', (e) => {
        state.settings.zones.leftCheek = e.target.checked;
        state.settings.zones.rightCheek = e.target.checked;
    });

    elements.zoneChin.addEventListener('change', (e) => {
        state.settings.zones.chin = e.target.checked;
    });
}

async function startMonitoring() {
    // Show loading state
    elements.welcomeContent.classList.add('hidden');
    elements.loadingState.classList.add('visible');

    // Initialize MediaPipe
    const mpInit = await initializeMediaPipe();
    if (!mpInit) {
        elements.loadingState.classList.remove('visible');
        elements.welcomeContent.classList.remove('hidden');
        alert('Failed to initialize AI models. Please refresh and try again.');
        return;
    }

    // Initialize camera
    const camInit = await initializeCamera();
    if (!camInit) {
        elements.loadingState.classList.remove('visible');
        elements.welcomeContent.classList.remove('hidden');
        return;
    }

    // Hide loading, show video
    elements.loadingState.classList.remove('visible');

    // Update state
    state.isRunning = true;
    state.startTime = Date.now();
    state.touchCount = 0;
    state.lastTouchTime = null;

    // Update UI
    elements.connectionStatus.classList.add('active');
    elements.statusText.textContent = 'Active';
    elements.detectionStatus.classList.add('visible');

    // Start detection loop
    detectFrame();
}

function stopMonitoring() {
    state.isRunning = false;

    // Cancel animation frame
    if (state.animationId) {
        cancelAnimationFrame(state.animationId);
        state.animationId = null;
    }

    // Stop camera
    if (elements.video.srcObject) {
        elements.video.srcObject.getTracks().forEach(track => track.stop());
        elements.video.srcObject = null;
    }

    // Clear canvas
    if (state.ctx) {
        state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    }

    // Update UI
    elements.connectionStatus.classList.remove('active');
    elements.statusText.textContent = 'Inactive';
    elements.detectionStatus.classList.remove('visible');
    elements.welcomeContent.classList.remove('hidden');
}

function resetStatistics() {
    state.touchCount = 0;
    state.lastTouchTime = null;
    state.startTime = Date.now();

    elements.touchCount.textContent = '0';
    elements.lastTouch.textContent = '--';
    elements.touchRate.textContent = '0.0';
}

// ============================================================================
// Initialize
// ============================================================================

setupEventListeners();
