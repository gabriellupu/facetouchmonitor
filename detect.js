/**
 * Face Touch Monitor - MediaPipe-powered face touch detection
 * Uses MediaPipe Face Mesh and Hand Landmarker for accurate proximity detection
 */

import { FaceLandmarker, HandLandmarker, PoseLandmarker, FilesetResolver, DrawingUtils } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs';

// ============================================================================
// State Management
// ============================================================================

const state = {
    isRunning: false,
    faceLandmarker: null,
    handLandmarker: null,
    poseLandmarker: null,
    video: null,
    canvas: null,
    ctx: null,
    drawingUtils: null,
    animationId: null,

    // Lateral camera for posture detection
    lateralVideo: null,
    lateralCanvas: null,
    lateralCtx: null,
    lateralStream: null,
    availableCameras: [],

    // Statistics
    touchCount: 0,
    lastTouchTime: null,
    postureAlertCount: 0,
    lastPostureAlertTime: null,
    startTime: null,
    lastFrameTime: 0,
    fps: 0,

    // Alert state
    alertCooldown: false,
    isTouching: false,
    wasTouching: false,
    continuousBeepTimer: null,

    // Posture alert state
    isBadPosture: false,
    wasBadPosture: false,
    postureBeepTimer: null,
    currentPostureIssue: null,

    // Audio context (lazy init)
    audioContext: null,

    // Settings
    settings: {
        sensitivity: 100,
        alertCooldownMs: 2000,
        soundEnabled: true,
        notifyEnabled: false,
        visualAlertEnabled: true,
        showFaceMesh: true,
        showHands: true,
        showProximity: false,
        // Front camera selection for nail biting detection
        frontCameraId: null,
        // Detection zones - which regions trigger alerts
        zones: {
            mouth: true,
            nose: true,
            leftEye: true,
            rightEye: true,
            leftCheek: false,
            rightCheek: false,
            chin: false
        },
        // Posture detection settings
        posture: {
            enabled: false,
            lateralCameraId: null,
            showPoseLandmarks: true,
            sensitivity: 100,
            // Fine-tuning thresholds (percentages that can be adjusted)
            headForwardThreshold: 15,      // degrees - head tilt forward
            shoulderSlouchThreshold: 10,    // degrees - shoulder drop
            spineAngleThreshold: 15,        // degrees - spine curvature
            alertCooldownMs: 3000           // separate cooldown for posture alerts
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

// LocalStorage key for settings persistence
const SETTINGS_STORAGE_KEY = 'faceTouchMonitor_settings';

// ============================================================================
// Settings Persistence
// ============================================================================

function saveSettings() {
    try {
        localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(state.settings));
    } catch (error) {
        console.warn('Failed to save settings to localStorage:', error);
    }
}

function loadSettings() {
    try {
        const saved = localStorage.getItem(SETTINGS_STORAGE_KEY);
        if (saved) {
            const parsed = JSON.parse(saved);
            // Merge with defaults to ensure all keys exist
            state.settings = { ...state.settings, ...parsed };
            // Handle nested zones object
            if (parsed.zones) {
                state.settings.zones = { ...state.settings.zones, ...parsed.zones };
            }
            // Handle nested posture object
            if (parsed.posture) {
                state.settings.posture = { ...state.settings.posture, ...parsed.posture };
            }
        }
    } catch (error) {
        console.warn('Failed to load settings from localStorage:', error);
    }
}

function applySettingsToUI() {
    // Alert toggles
    elements.beepToggle.checked = state.settings.soundEnabled;
    elements.notifyToggle.checked = state.settings.notifyEnabled;
    elements.visualAlertToggle.checked = state.settings.visualAlertEnabled;

    // Cooldown slider
    const cooldownSec = state.settings.alertCooldownMs / 1000;
    elements.alertCooldown.value = cooldownSec;
    elements.cooldownValue.textContent = `${cooldownSec}s`;

    // Visualization toggles
    elements.showLandmarks.checked = state.settings.showFaceMesh;
    elements.showHands.checked = state.settings.showHands;
    elements.showProximity.checked = state.settings.showProximity;

    // Sensitivity slider
    elements.sensitivitySlider.value = state.settings.sensitivity;
    elements.sensitivityValue.textContent = `${state.settings.sensitivity}%`;

    // Zone toggles
    elements.zoneMouth.checked = state.settings.zones.mouth;
    elements.zoneNose.checked = state.settings.zones.nose;
    elements.zoneEyes.checked = state.settings.zones.leftEye && state.settings.zones.rightEye;
    elements.zoneCheeks.checked = state.settings.zones.leftCheek && state.settings.zones.rightCheek;
    elements.zoneChin.checked = state.settings.zones.chin;

    // Posture settings
    if (elements.postureToggle) {
        elements.postureToggle.checked = state.settings.posture.enabled;
    }
    if (elements.showPoseLandmarks) {
        elements.showPoseLandmarks.checked = state.settings.posture.showPoseLandmarks;
    }
    if (elements.postureSensitivitySlider) {
        elements.postureSensitivitySlider.value = state.settings.posture.sensitivity;
        elements.postureSensitivityValue.textContent = `${state.settings.posture.sensitivity}%`;
    }
    if (elements.headForwardSlider) {
        elements.headForwardSlider.value = state.settings.posture.headForwardThreshold;
        elements.headForwardValue.textContent = `${state.settings.posture.headForwardThreshold}°`;
    }
    if (elements.shoulderSlouchSlider) {
        elements.shoulderSlouchSlider.value = state.settings.posture.shoulderSlouchThreshold;
        elements.shoulderSlouchValue.textContent = `${state.settings.posture.shoulderSlouchThreshold}°`;
    }
    if (elements.spineAngleSlider) {
        elements.spineAngleSlider.value = state.settings.posture.spineAngleThreshold;
        elements.spineAngleValue.textContent = `${state.settings.posture.spineAngleThreshold}°`;
    }
    if (elements.postureCooldownSlider) {
        const postureCooldownSec = state.settings.posture.alertCooldownMs / 1000;
        elements.postureCooldownSlider.value = postureCooldownSec;
        elements.postureCooldownValue.textContent = `${postureCooldownSec}s`;
    }

    // Update lateral video wrapper visibility
    updateLateralCameraVisibility();
}

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
    cooldownValue: document.getElementById('cooldownValue'),
    showLandmarks: document.getElementById('showLandmarks'),
    showHands: document.getElementById('showHands'),
    showProximity: document.getElementById('showProximity'),
    sensitivitySlider: document.getElementById('sensitivitySlider'),
    sensitivityValue: document.getElementById('sensitivityValue'),
    frontCameraSelect: document.getElementById('frontCameraSelect'),

    // Zone toggles
    zoneMouth: document.getElementById('zoneMouth'),
    zoneNose: document.getElementById('zoneNose'),
    zoneEyes: document.getElementById('zoneEyes'),
    zoneCheeks: document.getElementById('zoneCheeks'),
    zoneChin: document.getElementById('zoneChin'),

    // Posture detection elements
    postureToggle: document.getElementById('postureToggle'),
    lateralCameraSelect: document.getElementById('lateralCameraSelect'),
    lateralVideo: document.getElementById('lateralVideoElement'),
    lateralCanvas: document.getElementById('lateralOverlayCanvas'),
    lateralVideoWrapper: document.getElementById('lateralVideoWrapper'),
    showPoseLandmarks: document.getElementById('showPoseLandmarks'),
    postureSensitivitySlider: document.getElementById('postureSensitivitySlider'),
    postureSensitivityValue: document.getElementById('postureSensitivityValue'),
    headForwardSlider: document.getElementById('headForwardSlider'),
    headForwardValue: document.getElementById('headForwardValue'),
    shoulderSlouchSlider: document.getElementById('shoulderSlouchSlider'),
    shoulderSlouchValue: document.getElementById('shoulderSlouchValue'),
    spineAngleSlider: document.getElementById('spineAngleSlider'),
    spineAngleValue: document.getElementById('spineAngleValue'),
    postureCooldownSlider: document.getElementById('postureCooldownSlider'),
    postureCooldownValue: document.getElementById('postureCooldownValue'),
    postureAlertCount: document.getElementById('postureAlertCount'),
    lastPostureAlert: document.getElementById('lastPostureAlert'),
    postureStatus: document.getElementById('postureStatus')
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

        // Initialize Pose Landmarker for posture detection
        state.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        return true;
    } catch (error) {
        console.error('Failed to initialize MediaPipe:', error);
        return false;
    }
}

async function initializeCamera(deviceId = null) {
    try {
        // Stop existing stream if any
        if (elements.video.srcObject) {
            elements.video.srcObject.getTracks().forEach(track => track.stop());
        }

        // Build video constraints
        const videoConstraints = {
            width: { ideal: 1280 },
            height: { ideal: 720 }
        };

        // Use specific device if provided, otherwise use front-facing camera
        if (deviceId) {
            videoConstraints.deviceId = { exact: deviceId };
        } else {
            videoConstraints.facingMode = 'user';
        }

        const stream = await navigator.mediaDevices.getUserMedia({
            video: videoConstraints,
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
// Lateral Camera Management (for Posture Detection)
// ============================================================================

async function enumerateCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        state.availableCameras = devices.filter(device => device.kind === 'videoinput');
        populateCameraSelect();
        return state.availableCameras;
    } catch (error) {
        console.error('Failed to enumerate cameras:', error);
        return [];
    }
}

function populateCameraSelect() {
    // Populate front camera select
    if (elements.frontCameraSelect) {
        elements.frontCameraSelect.innerHTML = '<option value="">Default camera</option>';

        state.availableCameras.forEach((camera, index) => {
            const option = document.createElement('option');
            option.value = camera.deviceId;
            option.textContent = camera.label || `Camera ${index + 1}`;
            elements.frontCameraSelect.appendChild(option);
        });

        // Select previously chosen camera if available
        if (state.settings.frontCameraId) {
            elements.frontCameraSelect.value = state.settings.frontCameraId;
        }
    }

    // Populate lateral camera select
    if (elements.lateralCameraSelect) {
        elements.lateralCameraSelect.innerHTML = '<option value="">Select lateral camera...</option>';

        state.availableCameras.forEach((camera, index) => {
            const option = document.createElement('option');
            option.value = camera.deviceId;
            option.textContent = camera.label || `Camera ${index + 1}`;
            elements.lateralCameraSelect.appendChild(option);
        });

        // Select previously chosen camera if available
        if (state.settings.posture.lateralCameraId) {
            elements.lateralCameraSelect.value = state.settings.posture.lateralCameraId;
        }
    }
}

async function initializeLateralCamera(deviceId) {
    try {
        // Stop existing lateral stream if any
        stopLateralCamera();

        if (!deviceId) return false;

        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                deviceId: { exact: deviceId },
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });

        state.lateralStream = stream;
        elements.lateralVideo.srcObject = stream;

        return new Promise((resolve) => {
            elements.lateralVideo.onloadedmetadata = () => {
                elements.lateralVideo.play();

                // Set canvas dimensions to match video
                elements.lateralCanvas.width = elements.lateralVideo.videoWidth;
                elements.lateralCanvas.height = elements.lateralVideo.videoHeight;

                state.lateralVideo = elements.lateralVideo;
                state.lateralCanvas = elements.lateralCanvas;
                state.lateralCtx = elements.lateralCanvas.getContext('2d');

                console.log('Lateral camera initialized successfully');
                resolve(true);
            };
        });
    } catch (error) {
        console.error('Failed to initialize lateral camera:', error);
        alert('Unable to access the selected lateral camera. Please try another camera.');
        return false;
    }
}

function stopLateralCamera() {
    if (state.lateralStream) {
        state.lateralStream.getTracks().forEach(track => track.stop());
        state.lateralStream = null;
    }
    if (elements.lateralVideo) {
        elements.lateralVideo.srcObject = null;
    }
    state.lateralVideo = null;
    state.lateralCanvas = null;
    state.lateralCtx = null;
}

function updateLateralCameraVisibility() {
    if (elements.lateralVideoWrapper) {
        if (state.settings.posture.enabled && state.lateralStream) {
            elements.lateralVideoWrapper.classList.add('visible');
        } else {
            elements.lateralVideoWrapper.classList.remove('visible');
        }
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

    // Process posture detection from lateral camera
    if (state.settings.posture.enabled && state.lateralVideo && state.poseLandmarker) {
        const poseLandmarks = detectPostureFrame();
        const badPostureDetected = checkPosture(poseLandmarks);
        handlePostureState(badPostureDetected);
    }

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
// Posture Detection (using lateral camera)
// ============================================================================

// MediaPipe Pose Landmark indices for key body points
const POSE_LANDMARKS = {
    NOSE: 0,
    LEFT_EYE_INNER: 1,
    LEFT_EYE: 2,
    LEFT_EYE_OUTER: 3,
    RIGHT_EYE_INNER: 4,
    RIGHT_EYE: 5,
    RIGHT_EYE_OUTER: 6,
    LEFT_EAR: 7,
    RIGHT_EAR: 8,
    MOUTH_LEFT: 9,
    MOUTH_RIGHT: 10,
    LEFT_SHOULDER: 11,
    RIGHT_SHOULDER: 12,
    LEFT_ELBOW: 13,
    RIGHT_ELBOW: 14,
    LEFT_WRIST: 15,
    RIGHT_WRIST: 16,
    LEFT_PINKY: 17,
    RIGHT_PINKY: 18,
    LEFT_INDEX: 19,
    RIGHT_INDEX: 20,
    LEFT_THUMB: 21,
    RIGHT_THUMB: 22,
    LEFT_HIP: 23,
    RIGHT_HIP: 24,
    LEFT_KNEE: 25,
    RIGHT_KNEE: 26,
    LEFT_ANKLE: 27,
    RIGHT_ANKLE: 28,
    LEFT_HEEL: 29,
    RIGHT_HEEL: 30,
    LEFT_FOOT_INDEX: 31,
    RIGHT_FOOT_INDEX: 32
};

function checkPosture(poseLandmarks) {
    if (!poseLandmarks || !state.settings.posture.enabled) {
        state.currentPostureIssue = null;
        return false;
    }

    const issues = [];
    const sensitivity = state.settings.posture.sensitivity / 100;

    // Get key landmarks
    const nose = poseLandmarks[POSE_LANDMARKS.NOSE];
    const leftEar = poseLandmarks[POSE_LANDMARKS.LEFT_EAR];
    const rightEar = poseLandmarks[POSE_LANDMARKS.RIGHT_EAR];
    const leftShoulder = poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER];
    const rightShoulder = poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER];
    const leftHip = poseLandmarks[POSE_LANDMARKS.LEFT_HIP];
    const rightHip = poseLandmarks[POSE_LANDMARKS.RIGHT_HIP];

    // 1. Check head forward position (ear should be roughly above shoulder from side view)
    // In lateral view, if ear.x is significantly ahead of shoulder.x, head is forward
    const headForwardThreshold = state.settings.posture.headForwardThreshold / sensitivity;

    // Use the ear closest to camera (depending on which side the lateral camera is)
    const ear = leftEar.visibility > rightEar.visibility ? leftEar : rightEar;
    const shoulder = leftShoulder.visibility > rightShoulder.visibility ? leftShoulder : rightShoulder;

    if (ear && shoulder && ear.visibility > 0.5 && shoulder.visibility > 0.5) {
        // Calculate head forward angle
        const headForwardAngle = calculateAngle(
            { x: shoulder.x, y: shoulder.y - 0.1 }, // point above shoulder
            shoulder,
            ear
        );

        // If ear is significantly forward of shoulder (angle deviates from vertical)
        const headForwardDeviation = Math.abs(90 - headForwardAngle);
        if (headForwardDeviation > headForwardThreshold) {
            issues.push('Head forward');
        }
    }

    // 2. Check shoulder slouch (shoulders should be level)
    const shoulderSlouchThreshold = state.settings.posture.shoulderSlouchThreshold / sensitivity;

    if (leftShoulder && rightShoulder &&
        leftShoulder.visibility > 0.5 && rightShoulder.visibility > 0.5) {
        const shoulderAngle = Math.abs(
            Math.atan2(rightShoulder.y - leftShoulder.y, rightShoulder.x - leftShoulder.x) * (180 / Math.PI)
        );
        if (shoulderAngle > shoulderSlouchThreshold) {
            issues.push('Uneven shoulders');
        }
    }

    // 3. Check spine alignment (from side view - ear, shoulder, hip should be relatively aligned)
    const spineAngleThreshold = state.settings.posture.spineAngleThreshold / sensitivity;

    const hip = leftHip.visibility > rightHip.visibility ? leftHip : rightHip;

    if (ear && shoulder && hip &&
        ear.visibility > 0.5 && shoulder.visibility > 0.5 && hip.visibility > 0.5) {
        // Calculate the angle formed by ear-shoulder-hip
        const spineAngle = calculateAngle(ear, shoulder, hip);
        const spineDeviation = Math.abs(180 - spineAngle);

        if (spineDeviation > spineAngleThreshold) {
            issues.push('Spine curved');
        }
    }

    // 4. Check if shoulders are hunched forward (shoulder significantly ahead of hip in x)
    if (shoulder && hip && shoulder.visibility > 0.5 && hip.visibility > 0.5) {
        const hunchThreshold = 0.05 / sensitivity; // normalized coordinates
        if (shoulder.x - hip.x > hunchThreshold) {
            issues.push('Shoulders hunched');
        }
    }

    if (issues.length > 0) {
        state.currentPostureIssue = issues.join(', ');
        return true;
    }

    state.currentPostureIssue = null;
    return false;
}

function calculateAngle(pointA, pointB, pointC) {
    // Calculate angle at pointB formed by pointA-pointB-pointC
    const radians = Math.atan2(pointC.y - pointB.y, pointC.x - pointB.x) -
                    Math.atan2(pointA.y - pointB.y, pointA.x - pointB.x);
    let angle = Math.abs(radians * (180 / Math.PI));
    if (angle > 180) {
        angle = 360 - angle;
    }
    return angle;
}

function detectPostureFrame() {
    if (!state.isRunning || !state.settings.posture.enabled || !state.lateralVideo) {
        return null;
    }

    const now = performance.now();

    try {
        // Get pose landmarks from lateral camera
        const poseResults = state.poseLandmarker.detectForVideo(state.lateralVideo, now);
        const poseLandmarks = poseResults.landmarks?.[0] || null;

        // Clear lateral canvas
        if (state.lateralCtx) {
            state.lateralCtx.clearRect(0, 0, state.lateralCanvas.width, state.lateralCanvas.height);

            // Draw pose landmarks if enabled
            if (poseLandmarks && state.settings.posture.showPoseLandmarks) {
                drawPoseLandmarks(poseLandmarks);
            }
        }

        return poseLandmarks;
    } catch (error) {
        console.warn('Posture detection error:', error);
        return null;
    }
}

function drawPoseLandmarks(landmarks) {
    if (!state.lateralCtx || !landmarks) return;

    const ctx = state.lateralCtx;
    const width = state.lateralCanvas.width;
    const height = state.lateralCanvas.height;

    // Draw connections
    const connections = PoseLandmarker.POSE_CONNECTIONS;
    ctx.strokeStyle = state.isBadPosture ? 'rgba(239, 68, 68, 0.8)' : 'rgba(34, 197, 94, 0.8)';
    ctx.lineWidth = 2;

    for (const connection of connections) {
        const start = landmarks[connection.start];
        const end = landmarks[connection.end];

        if (start.visibility > 0.5 && end.visibility > 0.5) {
            ctx.beginPath();
            ctx.moveTo(start.x * width, start.y * height);
            ctx.lineTo(end.x * width, end.y * height);
            ctx.stroke();
        }
    }

    // Draw landmarks
    for (let i = 0; i < landmarks.length; i++) {
        const lm = landmarks[i];
        if (lm.visibility > 0.5) {
            const x = lm.x * width;
            const y = lm.y * height;

            // Highlight key posture points
            const isKeyPoint = [
                POSE_LANDMARKS.NOSE,
                POSE_LANDMARKS.LEFT_EAR,
                POSE_LANDMARKS.RIGHT_EAR,
                POSE_LANDMARKS.LEFT_SHOULDER,
                POSE_LANDMARKS.RIGHT_SHOULDER,
                POSE_LANDMARKS.LEFT_HIP,
                POSE_LANDMARKS.RIGHT_HIP
            ].includes(i);

            const radius = isKeyPoint ? 6 : 3;
            const color = isKeyPoint
                ? (state.isBadPosture ? 'rgba(239, 68, 68, 1)' : 'rgba(245, 158, 11, 1)')
                : 'rgba(34, 197, 94, 0.6)';

            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();
        }
    }
}

// ============================================================================
// Touch State Handling
// ============================================================================

function handleTouchState(touchDetected, faceVisible) {
    state.wasTouching = state.isTouching;
    state.isTouching = touchDetected;

    // Update status display
    updateDetectionStatus(faceVisible, touchDetected);

    // Handle touch state transitions
    if (touchDetected && !state.wasTouching) {
        // New touch started - trigger immediately and start continuous beep timer
        triggerAlert();
        startContinuousBeep();
    } else if (!touchDetected && state.wasTouching) {
        // Touch ended - stop continuous beep
        stopContinuousBeep();
    }
}

function startContinuousBeep() {
    // Clear any existing timer
    stopContinuousBeep();

    // Set up interval to beep continuously while touching
    state.continuousBeepTimer = setInterval(() => {
        if (state.isTouching) {
            triggerContinuousAlert();
        } else {
            stopContinuousBeep();
        }
    }, state.settings.alertCooldownMs);
}

function stopContinuousBeep() {
    if (state.continuousBeepTimer) {
        clearInterval(state.continuousBeepTimer);
        state.continuousBeepTimer = null;
    }
}

// ============================================================================
// Posture Alert Handling
// ============================================================================

function handlePostureState(badPostureDetected) {
    state.wasBadPosture = state.isBadPosture;
    state.isBadPosture = badPostureDetected;

    // Update posture status display
    updatePostureStatus(badPostureDetected);

    // Handle posture state transitions
    if (badPostureDetected && !state.wasBadPosture) {
        // New bad posture detected - trigger immediately and start continuous alert
        triggerPostureAlert();
        startPostureContinuousAlert();
    } else if (!badPostureDetected && state.wasBadPosture) {
        // Posture corrected - stop continuous alert
        stopPostureContinuousAlert();
    }
}

function startPostureContinuousAlert() {
    stopPostureContinuousAlert();

    state.postureBeepTimer = setInterval(() => {
        if (state.isBadPosture) {
            triggerPostureContinuousAlert();
        } else {
            stopPostureContinuousAlert();
        }
    }, state.settings.posture.alertCooldownMs);
}

function stopPostureContinuousAlert() {
    if (state.postureBeepTimer) {
        clearInterval(state.postureBeepTimer);
        state.postureBeepTimer = null;
    }
}

function triggerPostureAlert() {
    state.postureAlertCount++;
    state.lastPostureAlertTime = Date.now();

    const issue = state.currentPostureIssue || 'Poor posture';
    console.log(`Posture alert: ${issue}! Count: ${state.postureAlertCount}`);

    // Sound alert (different tone for posture - lower frequency)
    if (state.settings.soundEnabled) {
        playBeep(330, 200); // Lower frequency, slightly longer
    }

    // Visual alert (using same flash but could be different color)
    if (state.settings.visualAlertEnabled) {
        elements.alertFlash.classList.add('active', 'posture');
        setTimeout(() => {
            elements.alertFlash.classList.remove('active', 'posture');
        }, 400);
    }

    // Browser notification
    if (state.settings.notifyEnabled) {
        sendNotification(`Posture alert: ${issue}! Count: ${state.postureAlertCount}`);
    }
}

function triggerPostureContinuousAlert() {
    const issue = state.currentPostureIssue || 'Poor posture';

    if (state.settings.soundEnabled) {
        playBeep(330, 200);
    }

    if (state.settings.visualAlertEnabled) {
        elements.alertFlash.classList.add('active', 'posture');
        setTimeout(() => {
            elements.alertFlash.classList.remove('active', 'posture');
        }, 400);
    }

    if (state.settings.notifyEnabled) {
        sendNotification(`Still bad posture: ${issue}!`);
    }
}

function updatePostureStatus(badPosture) {
    if (!elements.postureStatus) return;

    const statusText = elements.postureStatus.querySelector('.posture-status-text');
    if (!statusText) return;

    elements.postureStatus.classList.remove('good', 'bad');

    if (badPosture) {
        statusText.textContent = state.currentPostureIssue || 'Bad posture';
        elements.postureStatus.classList.add('bad');
    } else {
        statusText.textContent = 'Good posture';
        elements.postureStatus.classList.add('good');
    }
}

function triggerContinuousAlert() {
    // Only play sound and visual during continuous touch (no count increment)
    const zoneName = state.lastTouchedZone ? getZoneDisplayName(state.lastTouchedZone) : 'Face';

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

    // Browser notification (optional for continuous - can be noisy)
    if (state.settings.notifyEnabled) {
        sendNotification(`Still touching ${zoneName}!`);
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

    // Update posture statistics
    if (elements.postureAlertCount) {
        elements.postureAlertCount.textContent = state.postureAlertCount;
    }
    if (elements.lastPostureAlert && state.lastPostureAlertTime) {
        const secAgo = Math.round((Date.now() - state.lastPostureAlertTime) / 1000);
        elements.lastPostureAlert.textContent = `${secAgo}s ago`;
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

function initAudioContext() {
    if (!state.audioContext) {
        state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    return state.audioContext;
}

async function playBeep(frequency, duration) {
    try {
        const ctx = initAudioContext();

        // Resume context if suspended (happens in background tabs)
        if (ctx.state === 'suspended') {
            await ctx.resume();
        }

        const oscillator = ctx.createOscillator();
        const gainNode = ctx.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(ctx.destination);

        oscillator.frequency.value = frequency;
        oscillator.type = 'sine';

        gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration / 1000);

        oscillator.start();
        oscillator.stop(ctx.currentTime + duration / 1000);
    } catch (error) {
        console.warn('Audio playback failed:', error);
    }
}

// Keep audio context alive when tab visibility changes
document.addEventListener('visibilitychange', () => {
    if (state.audioContext && state.audioContext.state === 'suspended' && state.isRunning) {
        state.audioContext.resume();
    }
});

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
        saveSettings();
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
        saveSettings();
    });

    elements.visualAlertToggle.addEventListener('change', (e) => {
        state.settings.visualAlertEnabled = e.target.checked;
        saveSettings();
    });

    // Cooldown slider
    elements.alertCooldown.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        state.settings.alertCooldownMs = value * 1000;
        elements.cooldownValue.textContent = `${value}s`;
        saveSettings();
    });

    // Visualization controls
    elements.showLandmarks.addEventListener('change', (e) => {
        state.settings.showFaceMesh = e.target.checked;
        saveSettings();
    });

    elements.showHands.addEventListener('change', (e) => {
        state.settings.showHands = e.target.checked;
        saveSettings();
    });

    elements.showProximity.addEventListener('change', (e) => {
        state.settings.showProximity = e.target.checked;
        saveSettings();
    });

    // Sensitivity slider
    elements.sensitivitySlider.addEventListener('input', (e) => {
        state.settings.sensitivity = parseInt(e.target.value, 10);
        elements.sensitivityValue.textContent = `${state.settings.sensitivity}%`;
        saveSettings();
    });

    // Front camera selection
    if (elements.frontCameraSelect) {
        elements.frontCameraSelect.addEventListener('change', async (e) => {
            const deviceId = e.target.value;
            state.settings.frontCameraId = deviceId || null;
            saveSettings();

            // If monitoring is running, switch cameras
            if (state.isRunning) {
                await initializeCamera(deviceId || null);
            }
        });
    }

    // Zone toggles
    elements.zoneMouth.addEventListener('change', (e) => {
        state.settings.zones.mouth = e.target.checked;
        saveSettings();
    });

    elements.zoneNose.addEventListener('change', (e) => {
        state.settings.zones.nose = e.target.checked;
        saveSettings();
    });

    elements.zoneEyes.addEventListener('change', (e) => {
        state.settings.zones.leftEye = e.target.checked;
        state.settings.zones.rightEye = e.target.checked;
        saveSettings();
    });

    elements.zoneCheeks.addEventListener('change', (e) => {
        state.settings.zones.leftCheek = e.target.checked;
        state.settings.zones.rightCheek = e.target.checked;
        saveSettings();
    });

    elements.zoneChin.addEventListener('change', (e) => {
        state.settings.zones.chin = e.target.checked;
        saveSettings();
    });

    // Posture detection controls
    if (elements.postureToggle) {
        elements.postureToggle.addEventListener('change', async (e) => {
            state.settings.posture.enabled = e.target.checked;
            saveSettings();

            if (e.target.checked) {
                // Enable posture detection - initialize lateral camera if selected
                if (state.settings.posture.lateralCameraId) {
                    await initializeLateralCamera(state.settings.posture.lateralCameraId);
                }
            } else {
                // Disable posture detection - stop lateral camera
                stopLateralCamera();
                stopPostureContinuousAlert();
                state.isBadPosture = false;
                state.wasBadPosture = false;
            }
            updateLateralCameraVisibility();
        });
    }

    if (elements.lateralCameraSelect) {
        elements.lateralCameraSelect.addEventListener('change', async (e) => {
            const deviceId = e.target.value;
            state.settings.posture.lateralCameraId = deviceId || null;
            saveSettings();

            if (deviceId && state.settings.posture.enabled) {
                await initializeLateralCamera(deviceId);
            } else {
                stopLateralCamera();
            }
            updateLateralCameraVisibility();
        });
    }

    if (elements.showPoseLandmarks) {
        elements.showPoseLandmarks.addEventListener('change', (e) => {
            state.settings.posture.showPoseLandmarks = e.target.checked;
            saveSettings();
        });
    }

    if (elements.postureSensitivitySlider) {
        elements.postureSensitivitySlider.addEventListener('input', (e) => {
            state.settings.posture.sensitivity = parseInt(e.target.value, 10);
            elements.postureSensitivityValue.textContent = `${state.settings.posture.sensitivity}%`;
            saveSettings();
        });
    }

    if (elements.headForwardSlider) {
        elements.headForwardSlider.addEventListener('input', (e) => {
            state.settings.posture.headForwardThreshold = parseInt(e.target.value, 10);
            elements.headForwardValue.textContent = `${state.settings.posture.headForwardThreshold}°`;
            saveSettings();
        });
    }

    if (elements.shoulderSlouchSlider) {
        elements.shoulderSlouchSlider.addEventListener('input', (e) => {
            state.settings.posture.shoulderSlouchThreshold = parseInt(e.target.value, 10);
            elements.shoulderSlouchValue.textContent = `${state.settings.posture.shoulderSlouchThreshold}°`;
            saveSettings();
        });
    }

    if (elements.spineAngleSlider) {
        elements.spineAngleSlider.addEventListener('input', (e) => {
            state.settings.posture.spineAngleThreshold = parseInt(e.target.value, 10);
            elements.spineAngleValue.textContent = `${state.settings.posture.spineAngleThreshold}°`;
            saveSettings();
        });
    }

    if (elements.postureCooldownSlider) {
        elements.postureCooldownSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            state.settings.posture.alertCooldownMs = value * 1000;
            elements.postureCooldownValue.textContent = `${value}s`;
            saveSettings();
        });
    }
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

    // Enumerate available cameras first (needed for camera selection)
    await enumerateCameras();

    // Initialize front camera (use selected camera if available)
    const camInit = await initializeCamera(state.settings.frontCameraId);
    if (!camInit) {
        elements.loadingState.classList.remove('visible');
        elements.welcomeContent.classList.remove('hidden');
        return;
    }

    // Initialize lateral camera if posture detection is enabled and camera is selected
    if (state.settings.posture.enabled && state.settings.posture.lateralCameraId) {
        await initializeLateralCamera(state.settings.posture.lateralCameraId);
    }

    // Hide loading, show video
    elements.loadingState.classList.remove('visible');

    // Update state
    state.isRunning = true;
    state.startTime = Date.now();
    state.touchCount = 0;
    state.lastTouchTime = null;
    state.postureAlertCount = 0;
    state.lastPostureAlertTime = null;

    // Update UI
    elements.connectionStatus.classList.add('active');
    elements.statusText.textContent = 'Active';
    elements.detectionStatus.classList.add('visible');
    updateLateralCameraVisibility();

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

    // Stop continuous beep timer
    stopContinuousBeep();

    // Stop posture continuous alert timer
    stopPostureContinuousAlert();

    // Stop main camera
    if (elements.video.srcObject) {
        elements.video.srcObject.getTracks().forEach(track => track.stop());
        elements.video.srcObject = null;
    }

    // Stop lateral camera
    stopLateralCamera();

    // Clear canvas
    if (state.ctx) {
        state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    }

    // Clear lateral canvas
    if (state.lateralCtx) {
        state.lateralCtx.clearRect(0, 0, state.lateralCanvas.width, state.lateralCanvas.height);
    }

    // Reset posture state
    state.isBadPosture = false;
    state.wasBadPosture = false;

    // Update UI
    elements.connectionStatus.classList.remove('active');
    elements.statusText.textContent = 'Inactive';
    elements.detectionStatus.classList.remove('visible');
    elements.welcomeContent.classList.remove('hidden');
    updateLateralCameraVisibility();
}

function resetStatistics() {
    state.touchCount = 0;
    state.lastTouchTime = null;
    state.postureAlertCount = 0;
    state.lastPostureAlertTime = null;
    state.startTime = Date.now();

    elements.touchCount.textContent = '0';
    elements.lastTouch.textContent = '--';
    elements.touchRate.textContent = '0.0';

    if (elements.postureAlertCount) {
        elements.postureAlertCount.textContent = '0';
    }
    if (elements.lastPostureAlert) {
        elements.lastPostureAlert.textContent = '--';
    }
}

// ============================================================================
// Initialize
// ============================================================================

// Load saved settings and apply to UI
loadSettings();
applySettingsToUI();

setupEventListeners();
