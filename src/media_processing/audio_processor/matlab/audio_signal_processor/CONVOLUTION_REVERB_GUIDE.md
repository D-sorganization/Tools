# Convolution Reverb Guide

## What is Convolution Reverb?

Convolution reverb uses **impulse responses (IRs)** to simulate acoustic spaces. An impulse response is a recording of how a real space responds to a short, sharp sound (like a balloon pop or starter pistol). When you convolve your audio with an IR, it sounds like your audio was recorded in that space!

This is how Hollywood creates convincing reverb - they use impulse responses from real locations.

## Quick Start

```matlab
% Create reverb processor
reverb = ConvolutionReverb();

% Use built-in IR
reverb.loadBuiltIn('concert_hall');

% Load your audio
[vocal, fs] = audioread('dry_vocal.wav');

% Apply reverb
wet = reverb.process(vocal, fs, 'WetDry', 0.3);  % 30% wet, 70% dry

% Save result
audiowrite('vocal_with_reverb.wav', wet, fs);
```

## Built-in Impulse Responses

The system includes synthetic IRs for common spaces:

```matlab
reverb.listAvailableIRs();  % Shows all built-in IRs
```

Available spaces:

- `small_room` - Bedroom, small studio (0.5s)
- `medium_room` - Living room (1.0s)
- `concert_hall` - Large concert hall (3.0s)
- `chamber` - Small booth/chamber (0.3s)
- `plate` - Classic plate reverb (2.0s)
- `spring` - Vintage spring reverb (1.5s)
- `ambience` - Subtle room ambience (0.8s)

## Using Real Impulse Responses

### Loading IR from File

```matlab
reverb = ConvolutionReverb();
reverb.loadIR('path/to/your_ir.wav');  % Load real IR
processed = reverb.process(audio, fs);
```

### Where to Get Real IRs

**Free IR Libraries:**

1. **OpenAIR** (University of York)
   - URL: https://www.openair.hosted.york.ac.uk/
   - 100+ free IRs from real spaces
   - Concert halls, churches, studios, stairwells, etc.

2. **EchoThief** (Audioease)
   - URL: http://www.echothief.com/
   - Free IRs from famous spaces
   - Very high quality

3. **Voxengo Impulses**
   - Free IR library
   - Various spaces and vintage hardware

4. **ReverbIR**
   - Community-driven IR collection

**Recording Your Own IRs:**

You can record your own impulse responses:

1. Generate a sine sweep (20Hz-20kHz) in your DAW
2. Play it through a speaker in the space you want to capture
3. Record the sweep with a good microphone
4. Use deconvolution software to extract the IR
   - Free tools: Altiverb, REW (Room EQ Wizard)

```matlab
% Example: Generate sine sweep for IR recording
fs = 44100;
duration = 5;  % seconds
t = linspace(0, duration, duration * fs);
f0 = 20;
f1 = 20000;
sweep = sin(2*pi * f0 * duration/log(f1/f0) * (exp(t*log(f1/f0)/duration) - 1));
audiowrite('sweep.wav', sweep, fs);
```

## Complete Examples

### Example 1: Compare Different Spaces

```matlab
reverb = ConvolutionReverb();
[audio, fs] = audioread('drums.wav');

% Try different spaces
spaces = {'small_room', 'medium_room', 'concert_hall', 'chamber'};

for i = 1:length(spaces)
    reverb.loadBuiltIn(spaces{i});
    result = reverb.process(audio, fs, 'WetDry', 0.4);
    audiowrite(sprintf('drums_%s.wav', spaces{i}), result, fs);
end
```

### Example 2: Vocal Reverb with Pre-Delay

```matlab
reverb = ConvolutionReverb();
reverb.loadBuiltIn('plate');

% Pre-delay keeps vocal clear before reverb kicks in
reverb.setPreDelay(0.05);  % 50ms pre-delay

% Subtle reverb
reverb.setWetDry(0.25, 0.75);  % 25% wet, 75% dry

[vocal, fs] = audioread('vocal.wav');
processed = reverb.process(vocal, fs);
```

### Example 3: Custom Reverb with EQ and Damping

```matlab
reverb = ConvolutionReverb();
reverb.loadBuiltIn('concert_hall');

% Dark, lush reverb
reverb.setWetDry(0.35, 0.65);
reverb.setPreDelay(0.03);
reverb.setEQ(-2, 0, -6);  % Cut low and high in reverb
reverb.setDamping(0.4);   % Roll off highs over time
reverb.setStereoWidth(1.5);  % Wider stereo image

[audio, fs] = audioread('synth.wav');
processed = reverb.process(audio, fs);
```

### Example 4: Reverse Reverb Effect

```matlab
reverb = ConvolutionReverb();
reverb.loadBuiltIn('concert_hall');
reverb.reverseIR();  % Reverse the impulse response!

[audio, fs] = audioread('snare.wav');
reversed = reverb.process(audio, fs, 'WetDry', 0.6);

% Creates swelling reverb that builds up to the hit
```

### Example 5: Use Real IR from OpenAIR

```matlab
% After downloading IR from openair.hosted.york.ac.uk
reverb = ConvolutionReverb();
reverb.loadIR('maes_howe_true_stereo.wav');  % Ancient tomb in Scotland!

reverb.setWetDry(0.5, 0.5);
[audio, fs] = audioread('voice.wav');
ancient_sound = reverb.process(audio, fs);
```

### Example 6: Truncate Long Reverb Tails

```matlab
reverb = ConvolutionReverb();
reverb.loadBuiltIn('concert_hall');

% Truncate reverb to 1.5 seconds (useful for dense mixes)
reverb.setTailLength(1.5);

[audio, fs] = audioread('guitar.wav');
processed = reverb.process(audio, fs);
```

## Advanced Control

### Wet/Dry Mix

Control the balance between original (dry) and reverb (wet):

```matlab
% Subtle ambience
reverb.setWetDry(0.15, 0.85);  % 15% reverb

% Huge reverb
reverb.setWetDry(0.6, 0.4);  % 60% reverb

% 100% wet (no dry signal - useful for parallel processing)
reverb.setWetDry(1.0, 0.0);
```

### Pre-Delay

Pre-delay adds a gap before reverb starts, keeping the dry signal clear:

```matlab
% No pre-delay (immediate reverb)
reverb.setPreDelay(0);

% Short pre-delay (keeps vocals clear)
reverb.setPreDelay(0.05);  % 50ms

% Long pre-delay (special effect)
reverb.setPreDelay(0.15);  % 150ms
```

### EQ on Reverb

Shape the tone of the reverb without affecting the dry signal:

```matlab
% Dark reverb (cut highs)
reverb.setEQ(0, 0, -8);

% Bright reverb (boost highs)
reverb.setEQ(0, 0, +4);

% Thin reverb (cut lows)
reverb.setEQ(-6, 0, 0);

% Warm reverb (boost lows, cut highs)
reverb.setEQ(+3, 0, -4);
```

### Damping

Simulates air absorption - high frequencies decay faster:

```matlab
% No damping (unnatural but clear)
reverb.setDamping(0);

% Natural damping
reverb.setDamping(0.3);

% Heavy damping (muffled, distant)
reverb.setDamping(0.7);
```

### Stereo Width

Control the stereo image of the reverb:

```matlab
% Mono reverb
reverb.setStereoWidth(0);

% Normal stereo
reverb.setStereoWidth(1.0);

% Extra wide (immersive)
reverb.setStereoWidth(1.5);
```

## IR Manipulation

### Trim Silent Parts

```matlab
reverb.loadIR('long_ir_with_silence.wav');
reverb.trimIR();  % Removes silence from beginning and end
```

### Normalize IR

```matlab
reverb.loadIR('quiet_ir.wav');
reverb.normalizeIR();  % Normalize to peak = 1.0
```

### Analyze IR

```matlab
reverb.loadIR('mystery_space.wav');
reverb.getIRInfo();  % Shows length, sample rate, peak level, etc.
reverb.plotIR();     % Plots waveform, envelope, frequency response
```

## Integration with Audio Processor

### Adding to AudioEffects

You can integrate convolution reverb into your existing audio processor:

```matlab
% In AudioEffects.m, add convolution reverb
effects = AudioEffects();
effects.ConvolutionReverb = ConvolutionReverb();

% In processing pipeline
function output = applyEffect(effects, audio, fs, effectName, params)
    switch effectName
        case 'ConvolutionReverb'
            effects.ConvolutionReverb.loadBuiltIn(params.space);
            effects.ConvolutionReverb.setWetDry(params.wet, params.dry);
            output = effects.ConvolutionReverb.process(audio, fs);
        % ... other effects
    end
end
```

### Parallel Processing

For professional mixing, use parallel reverb:

```matlab
reverb = ConvolutionReverb();
reverb.loadBuiltIn('plate');
reverb.setWetDry(1.0, 0.0);  % 100% wet

[audio, fs] = audioread('drums.wav');

% Process reverb
wetOnly = reverb.process(audio, fs);

% Mix manually (gives more control)
dryGain = 0.8;
wetGain = 0.3;
parallel = dryGain * audio + wetGain * wetOnly;
```

## Tips for Best Results

### 1. **Match Your Space to Your Source**

- **Vocals**: Plate, chamber, or small room
- **Drums**: Big hall, room
- **Guitar**: Spring, plate
- **Orchestral**: Large hall
- **Sound design**: Unusual spaces (stairwells, tunnels)

### 2. **Less is Often More**

Start with 20-30% wet mix and adjust from there. Too much reverb makes mixes muddy.

### 3. **Use Pre-Delay for Clarity**

Pre-delay (20-80ms) keeps the dry signal distinct, especially important for vocals.

### 4. **EQ Your Reverb**

Cut lows (below 200Hz) from reverb to prevent muddiness:

```matlab
reverb.setEQ(-6, 0, 0);  % Cut lows
```

### 5. **Truncate Long Tails**

Long reverb tails can clutter a mix. Truncate to 1-2 seconds for cleaner results:

```matlab
reverb.setTailLength(1.5);
```

### 6. **Use Different Reverbs for Different Elements**

Don't use the same reverb on everything:

```matlab
% Vocals - plate
vocalReverb = ConvolutionReverb();
vocalReverb.loadBuiltIn('plate');

% Drums - room
drumReverb = ConvolutionReverb();
drumReverb.loadBuiltIn('medium_room');

% Lead - hall
leadReverb = ConvolutionReverb();
leadReverb.loadBuiltIn('concert_hall');
```

## Comparison: Algorithmic vs Convolution Reverb

Your audio processor likely has algorithmic reverb (`AudioEffects`). Here's when to use each:

| Feature               | Algorithmic                  | Convolution          |
| --------------------- | ---------------------------- | -------------------- |
| **Sound**             | Synthetic, smooth            | Realistic, authentic |
| **CPU Usage**         | Low                          | High                 |
| **Real-time Control** | Easy                         | Limited              |
| **Best For**          | General use, live processing | Final mixes, realism |
| **Spaces**            | Generic                      | Specific real spaces |

**Use algorithmic reverb** when:

- You need low CPU usage
- You want to tweak parameters in real-time
- You need a generic "good" reverb

**Use convolution reverb** when:

- You need realism
- You want a specific space (e.g., "Sydney Opera House")
- You're doing final mixing
- You have real IRs from the space you want

## Troubleshooting

### Problem: Reverb sounds weak

**Solution**: Increase wet level or normalize the IR:

```matlab
reverb.normalizeIR();
reverb.setWetDry(0.4, 0.6);
```

### Problem: Reverb is too muddy

**Solution**: EQ out the lows, add pre-delay:

```matlab
reverb.setEQ(-6, 0, 0);  % Cut lows
reverb.setPreDelay(0.05);
reverb.setTailLength(1.2);  % Shorter tail
```

### Problem: Reverb sounds unnatural

**Solution**: Add damping and reduce stereo width:

```matlab
reverb.setDamping(0.4);
reverb.setStereoWidth(0.8);
```

### Problem: Output is clipping

The processor automatically normalizes, but if you still clip:

```matlab
% Reduce wet level
reverb.setWetDry(0.25, 0.75);

% Or normalize afterward
result = reverb.process(audio, fs);
result = result * 0.8;  % Reduce level
```

### Problem: Processing is slow

**Solution**: Truncate the IR, or use a shorter IR:

```matlab
reverb.setTailLength(2.0);  % Max 2 seconds
% Or use a shorter space
reverb.loadBuiltIn('chamber');  % Only 0.3 seconds
```

## Creative Uses

### 1. Reverse Reverb Swells

```matlab
reverb.loadBuiltIn('concert_hall');
reverb.reverseIR();
reversed = reverb.process(snare, fs, 'WetDry', 0.8);
```

### 2. Gated Reverb (80s drums)

```matlab
reverb.loadBuiltIn('medium_room');
reverb.setTailLength(0.3);  % Short, punchy
reverb.setWetDry(0.6, 0.4);
gated = reverb.process(drums, fs);
```

### 3. Infinite Reverb

```matlab
% Use a very long IR and 100% wet
reverb.loadBuiltIn('concert_hall');
reverb.setWetDry(0.95, 0.05);  % Almost all wet
infinite = reverb.process(pad, fs);
```

### 4. Convolution with Non-Space IRs

You can convolve with _anything_, not just room IRs:

```matlab
% Use a musical note as an "IR"
[note, fs] = audioread('guitar_chord.wav');
reverb.IR = note;
reverb.IRSampleRate = fs;

% Now convolution creates harmonic resonance!
weird = reverb.process(drums, fs);
```

## Performance Considerations

Convolution is CPU-intensive. The processing time depends on:

- **IR length**: Longer IRs = more processing
- **Audio length**: Longer audio = more processing
- **Sample rate**: Higher sample rates = more processing

**Optimization tips:**

1. **Truncate IRs**: Most reverb energy is in the first 1-2 seconds
2. **Downsample if possible**: If your source is 48kHz, consider processing at 44.1kHz
3. **Process in batches**: Process multiple files in sequence rather than loading IRs repeatedly

```matlab
% Good: Batch processing
reverb = ConvolutionReverb();
reverb.loadBuiltIn('concert_hall');  % Load once

files = {'track1.wav', 'track2.wav', 'track3.wav'};
for i = 1:length(files)
    [audio, fs] = audioread(files{i});
    processed = reverb.process(audio, fs);
    audiowrite(['processed_' files{i}], processed, fs);
end
```

## Summary

Convolution reverb is your tool for creating realistic acoustic spaces. Whether you need a subtle room ambience or the grandeur of a cathedral, impulse responses capture the true character of real spaces.

**Quick reference:**

```matlab
% Basic usage
reverb = ConvolutionReverb();
reverb.loadBuiltIn('concert_hall');  % or loadIR('myspace.wav')
result = reverb.process(audio, fs, 'WetDry', 0.3);

% Common adjustments
reverb.setWetDry(wet, dry);
reverb.setPreDelay(seconds);
reverb.setEQ(low, mid, high);
reverb.setDamping(amount);

% Analysis
reverb.getIRInfo();
reverb.plotIR();
```

Now go make your audio sound like it's in amazing spaces! 🎵
