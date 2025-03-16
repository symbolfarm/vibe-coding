# Beat Sequencer

A simple browser-based drum machine that lets you create and play beat patterns.

## Features

- Create drum patterns with kick, snare, hi-hat, and clap sounds
- Adjustable tempo (60-200 BPM)
- Play/stop functionality
- Clear pattern button
- Responsive design

## How to Use

1. Open `index.html` in a web browser
2. Click on the grid cells to activate/deactivate sounds
3. Adjust the tempo using the slider if desired
4. Click "Play" to start the sequence
5. Click "Clear" to reset the pattern

## Technical Details

This app uses the Web Audio API to generate and play drum sounds. The sequencer runs on a JavaScript timer that triggers sounds based on the active grid cells.