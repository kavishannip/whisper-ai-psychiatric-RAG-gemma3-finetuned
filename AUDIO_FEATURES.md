# Audio Features Documentation - Whisper AI-Psychiatric

## Overview
The Whisper AI-Psychiatric application now includes speech-to-text and text-to-speech capabilities to enhance user interaction through voice input and audio responses.

## Features Added

### 🎤 Speech-to-Text (STT)
- **Model**: Whisper-tiny (located in `stt-model/whisper-tiny/`)
- **Functionality**: Converts user voice input to text for chat interaction
- **Input Methods**:
  - Real-time audio recording (using microphone)
  - Audio file upload (supports WAV, MP3, M4A, FLAC)

### 🔊 Text-to-Speech (TTS)
- **Model**: Kokoro-82M (located in `tts-model/Kokoro-82M/`)
- **Functionality**: Converts AI responses to speech audio
- **Features**:
  - Adjustable speech speed (0.5x to 2.0x)
  - Auto-play option for responses
  - Manual play button for each response

## Installation Requirements

### Required Packages
Run one of the following to install audio processing packages:

**Option 1: Using batch file (Windows)**
```bash
install_audio_packages.bat
```

**Option 2: Using PowerShell (Windows)**
```powershell
.\install_audio_packages.ps1
```

**Option 3: Manual installation**
```bash
pip install librosa>=0.10.0
pip install soundfile>=0.12.0
pip install audio-recorder-streamlit>=0.0.8
pip install scipy>=1.9.0
```

### Updated requirements.txt
The requirements.txt file has been updated to include:
- `librosa>=0.10.0` - Audio processing library
- `soundfile>=0.12.0` - Audio file I/O
- `audio-recorder-streamlit>=0.0.8` - Streamlit audio recording component
- `scipy>=1.9.0` - Scientific computing (audio processing support)

## Usage Guide

### Using Speech-to-Text

1. **Real-time Recording**:
   - Click the microphone icon in the "Voice Input" section
   - Speak your question clearly
   - Click "Stop" when finished
   - Click "🔄 Transcribe Audio" to convert speech to text
   - The transcribed text will automatically be sent to the chat

2. **File Upload**:
   - If the microphone recorder is not available, use the file uploader
   - Upload an audio file (WAV, MP3, M4A, FLAC)
   - Click "🔄 Transcribe Uploaded Audio"
   - The transcribed text will be processed

### Using Text-to-Speech

1. **Enable/Disable TTS**:
   - Use the "Enable Text-to-Speech" checkbox in the sidebar
   - Adjust "Audio Speed" slider (0.5x to 2.0x normal speed)

2. **Playing Responses**:
   - Each AI response will have a "🔊 Play" button
   - Click to generate and play the audio version of the response
   - Audio will auto-play when generated

## Technical Implementation

### Speech-to-Text Pipeline
1. Audio input captured/uploaded
2. Audio processed using librosa (resampled to 16kHz)
3. Whisper model processes audio features
4. Generated transcription added to chat

### Text-to-Speech Pipeline
1. AI response text processed
2. Kokoro-82M model generates speech audio
3. Audio served through HTML5 audio player
4. Supports speed adjustment and auto-play

## Sidebar Features

### Model Status Indicators
- ✅ Whisper AI Model Loaded
- ✅ FAISS Index Loaded  
- ✅ Speech-to-Text Loaded

### Audio Settings
- **Enable Text-to-Speech**: Toggle TTS functionality
- **Audio Speed**: Adjust playback speed (0.5x - 2.0x)

### Voice Input Tips
- Speak clearly and distinctly
- Minimize background noise
- Keep recordings under 30 seconds for best results
- Ensure good microphone quality

## Troubleshooting

### Common Issues

1. **Microphone Not Working**:
   - Check browser permissions for microphone access
   - Use the file upload option as fallback
   - Ensure audio-recorder-streamlit is properly installed

2. **Audio Quality Issues**:
   - Use a quiet environment
   - Speak clearly and at normal pace
   - Check microphone quality

3. **TTS Not Working**:
   - Verify Kokoro-82M model is in correct directory
   - Check audio player compatibility in browser
   - Ensure scipy and audio libraries are installed

4. **Import Errors**:
   - Run the installation scripts
   - Manually install missing packages
   - Check virtual environment activation

### Model Paths
Ensure the following model directories exist:
- Speech-to-Text: `stt-model/whisper-tiny/`
- Text-to-Speech: `tts-model/Kokoro-82M/`
- Main AI Model: `model/Whisper-psychology-gemma-3-1b/`

## Browser Compatibility

### Recommended Browsers
- Chrome (best support for audio features)
- Firefox
- Edge
- Safari (may have limited microphone support)

### Required Permissions
- Microphone access for voice recording
- Audio playback for TTS responses

## Future Enhancements

### Planned Features
- Voice activity detection for hands-free operation
- Multiple voice options for TTS
- Real-time streaming transcription
- Noise cancellation for better STT accuracy
- Custom wake words for voice activation

### Performance Optimizations
- Model quantization for faster inference
- Audio preprocessing optimization
- Caching for frequently used TTS phrases
- Background audio processing

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Test with simple audio files first
4. Check browser console for error messages

## Version Information
- **Version**: 2.0 (Audio Features)
- **Added**: Speech-to-Text and Text-to-Speech capabilities
- **Base Version**: 1.0 (Text-only chat interface)
