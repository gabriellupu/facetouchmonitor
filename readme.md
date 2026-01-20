# Nail Biting Detector

An AI-powered web app that helps you stop nail biting by detecting when your hands approach your mouth and alerting you instantly.

**Try it live at [nailbitingdetector.com](https://nailbitingdetector.com)**

## Features

- **Real-time Detection** - Uses MediaPipe Face Mesh (468 landmarks) and Hand Landmarker (21 landmarks per hand) for precise tracking
- **Configurable Detection Zones** - Choose which areas trigger alerts: mouth, nose, eyes, cheeks, or chin
- **Multiple Alert Types** - Sound beeps, visual flash, and browser notifications
- **Privacy First** - All processing happens locally in your browser. No data is ever transmitted or recorded
- **60+ FPS Performance** - GPU-accelerated detection for smooth, responsive monitoring
- **Adjustable Sensitivity** - Fine-tune detection threshold to reduce false positives
- **Statistics Tracking** - Track touch count, rate per hour, and time since last touch

## How It Works

The app uses Google's [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) vision models to:

1. **Detect your face** with 468 precise facial landmarks
2. **Track your hands** with 21 landmarks per hand, including fingertips
3. **Calculate proximity** between fingertips and face regions
4. **Trigger alerts** when hands get too close to enabled detection zones

The detection uses z-depth checking to reduce false positives from hands that are in front of (but not touching) the face.

## Detection Zones

You can enable/disable specific face regions for detection:

| Zone | Color | Default | Use Case |
|------|-------|---------|----------|
| Mouth | Red | On | Nail biting detection |
| Nose | Orange | On | Face touching |
| Eyes | Blue | On | Eye rubbing |
| Cheeks | Purple | Off | Optional |
| Chin | Green | Off | Optional |

## Browser Support

- **Chrome** - Best support (recommended)
- **Firefox** - Full support
- **Edge** - Full support
- **Safari** - Limited support (WebGPU required)

Works on desktop and mobile devices with a front-facing camera.

## Tech Stack

- **MediaPipe Tasks Vision** - Face and hand landmark detection
- **Vanilla JavaScript** - ES6 modules, no framework dependencies
- **CSS Custom Properties** - Modern, responsive design
- **Web Audio API** - Sound alerts
- **Notifications API** - Browser notifications

## Local Development

No build process required. Simply serve the files with any static file server:

```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx serve

# Using PHP
php -S localhost:8000
```

Then open `http://localhost:8000` in your browser.

## Privacy

This app is designed with privacy as a core principle:

- All AI processing runs locally in your browser
- Your camera feed is never recorded or transmitted
- No analytics or tracking (you can verify in the source code)
- No external API calls except for loading the MediaPipe models from CDN

## Contributing

Pull requests are welcome! Please feel free to submit issues or feature requests.

## License

MIT License - feel free to use this code for your own projects.

## Author

Made by [@gabriellupu](https://github.com/gabriellupu)

---

*Originally forked from [webrtcHacks/facetouchmonitor](https://github.com/webrtchacks/facetouchmonitor) and modernized with MediaPipe.*
