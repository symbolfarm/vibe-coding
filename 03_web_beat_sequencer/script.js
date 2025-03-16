// Audio context
let audioContext;
let isPlaying = false;
let currentBeat = 0;
let tempo = 120;
let intervalId = null;

// Drum sounds
const sounds = {
    'Kick': null,
    'Snare': null,
    'Hi-hat': null,
    'Clap': null
};

// Sequencer grid dimensions
const rows = Object.keys(sounds);
const beats = 16; // 16th notes in a 4/4 measure

// DOM elements
const grid = document.getElementById('sequencer-grid');
const playButton = document.getElementById('play-button');
const clearButton = document.getElementById('clear-button');
const tempoSlider = document.getElementById('tempo');
const tempoValue = document.getElementById('tempo-value');

// Initialize audio context on user interaction
function initAudio() {
    if (audioContext) return;
    
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    // Load drum samples
    loadSample('kick.wav').then(buffer => sounds['Kick'] = buffer);
    loadSample('snare.wav').then(buffer => sounds['Snare'] = buffer);
    loadSample('hihat.wav').then(buffer => sounds['Hi-hat'] = buffer);
    loadSample('clap.wav').then(buffer => sounds['Clap'] = buffer);
    
    // Fallback: create basic sounds in case samples don't load
    createFallbackSounds();
}

// Load audio sample
async function loadSample(url) {
    try {
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        return await audioContext.decodeAudioData(arrayBuffer);
    } catch (error) {
        console.error('Error loading sample:', error);
        return null;
    }
}

// Create basic synthesized sounds as fallback
function createFallbackSounds() {
    // We'll create these sounds if the samples don't load
    setTimeout(() => {
        if (!sounds['Kick']) sounds['Kick'] = createKickSound();
        if (!sounds['Snare']) sounds['Snare'] = createSnareSound();
        if (!sounds['Hi-hat']) sounds['Hi-hat'] = createHiHatSound();
        if (!sounds['Clap']) sounds['Clap'] = createClapSound();
    }, 1000);
}

// Create synthesized kick drum sound
function createKickSound() {
    const bufferSize = audioContext.sampleRate * 0.5; // 500ms buffer
    const buffer = audioContext.createBuffer(1, bufferSize, audioContext.sampleRate);
    const data = buffer.getChannelData(0);
    
    let freq = 150;
    for (let i = 0; i < bufferSize; i++) {
        if (i > 20000) break; // Short sound
        const t = i / audioContext.sampleRate;
        data[i] = Math.sin(2 * Math.PI * freq * t) * Math.exp(-5 * t);
        freq *= 0.9992; // Frequency drop for kick sound
    }
    
    return buffer;
}

// Create synthesized snare drum sound
function createSnareSound() {
    const bufferSize = audioContext.sampleRate * 0.3;
    const buffer = audioContext.createBuffer(1, bufferSize, audioContext.sampleRate);
    const data = buffer.getChannelData(0);
    
    for (let i = 0; i < bufferSize; i++) {
        if (i > 10000) break;
        const t = i / audioContext.sampleRate;
        
        // Drum body (sine wave with fast decay)
        const body = Math.sin(2 * Math.PI * 180 * t) * Math.exp(-25 * t);
        
        // Reduced noise component with faster decay
        const white = Math.random() * 2 - 1;
        const noise = white * Math.exp(-15 * t) * 0.3;
        
        // Combine components
        data[i] = body + noise;
    }
    
    return buffer;
}

// Create synthesized hi-hat sound
function createHiHatSound() {
    const bufferSize = audioContext.sampleRate * 0.1;
    const buffer = audioContext.createBuffer(1, bufferSize, audioContext.sampleRate);
    const data = buffer.getChannelData(0);
    
    for (let i = 0; i < bufferSize; i++) {
        if (i > 5000) break;
        const t = i / audioContext.sampleRate;
        const white = Math.random() * 2 - 1;
        data[i] = white * Math.exp(-30 * t);
    }
    
    return buffer;
}

// Create synthesized clap sound
function createClapSound() {
    const bufferSize = audioContext.sampleRate * 0.2;
    const buffer = audioContext.createBuffer(1, bufferSize, audioContext.sampleRate);
    const data = buffer.getChannelData(0);
    
    for (let i = 0; i < bufferSize; i++) {
        if (i > 5000) break; // Short sound
        const t = i / audioContext.sampleRate;
        
        // Very fast attack and decay - more like a single hand clap
        let envelope;
        if (t < 0.001) {
            // Fast attack
            envelope = t * 1000; // Linear ramp up
        } else {
            // Fast decay
            envelope = Math.exp(-40 * t);
        }
        
        // Add sharp transient at beginning
        const transient = (t < 0.002) ? 0.8 : 0;
        
        // Small amount of noise for texture
        const noise = Math.random() * 0.2;
        
        // Combine with midrange frequency component for "slap" character
        const midFreq = 0.6 * Math.sin(2 * Math.PI * 800 * t);
        
        data[i] = (transient + midFreq + noise) * envelope * 0.7;
    }
    
    return buffer;
}

// Play a sound
function playSound(buffer) {
    if (!buffer || !audioContext) return;
    
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.start(0);
}

// Create sequencer grid
function createGrid() {
    grid.innerHTML = '';
    
    for (const row of rows) {
        const rowElement = document.createElement('div');
        rowElement.className = 'row';
        
        // Add row label
        const label = document.createElement('div');
        label.className = 'row-label';
        label.textContent = row;
        grid.appendChild(label);
        
        for (let beat = 0; beat < beats; beat++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.row = row;
            cell.dataset.beat = beat;
            
            cell.addEventListener('click', () => {
                cell.classList.toggle('active');
                
                // Play sound when cell is activated
                if (cell.classList.contains('active') && sounds[row]) {
                    initAudio();
                    playSound(sounds[row]);
                }
            });
            
            grid.appendChild(cell);
        }
    }
}

// Play sequence
function playSequence() {
    const beatTime = 60000 / tempo / 4; // ms per 16th note
    
    intervalId = setInterval(() => {
        // Remove 'playing' class from all cells
        document.querySelectorAll('.cell.playing').forEach(cell => {
            cell.classList.remove('playing');
        });
        
        // Get current beat cells
        const currentBeatCells = document.querySelectorAll(`.cell[data-beat="${currentBeat}"]`);
        
        // Add 'playing' class to current beat cells
        currentBeatCells.forEach(cell => {
            cell.classList.add('playing');
            
            // Play sound if cell is active
            if (cell.classList.contains('active')) {
                const row = cell.dataset.row;
                if (sounds[row]) {
                    playSound(sounds[row]);
                }
            }
        });
        
        // Advance beat
        currentBeat = (currentBeat + 1) % beats;
    }, beatTime);
}

// Stop sequence
function stopSequence() {
    clearInterval(intervalId);
    
    // Remove 'playing' class from all cells
    document.querySelectorAll('.cell.playing').forEach(cell => {
        cell.classList.remove('playing');
    });
}

// Toggle play/stop
function togglePlay() {
    initAudio();
    
    isPlaying = !isPlaying;
    
    if (isPlaying) {
        currentBeat = 0;
        playButton.textContent = 'Stop';
        playButton.classList.add('active');
        playSequence();
    } else {
        playButton.textContent = 'Play';
        playButton.classList.remove('active');
        stopSequence();
    }
}

// Clear all active cells
function clearGrid() {
    document.querySelectorAll('.cell.active').forEach(cell => {
        cell.classList.remove('active');
    });
}

// Initialize app
function init() {
    // Create sequencer grid
    createGrid();
    
    // Event listeners
    playButton.addEventListener('click', togglePlay);
    clearButton.addEventListener('click', clearGrid);
    
    // Tempo control
    tempoSlider.addEventListener('input', () => {
        tempo = parseInt(tempoSlider.value);
        tempoValue.textContent = tempo;
        
        // Update timing if playing
        if (isPlaying) {
            stopSequence();
            playSequence();
        }
    });
    
    // Handle first user interaction to initialize audio
    document.addEventListener('click', initAudio, { once: true });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);