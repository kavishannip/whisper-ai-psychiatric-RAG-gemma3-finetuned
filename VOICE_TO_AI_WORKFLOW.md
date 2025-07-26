# Voice-to-AI Workflow Documentation

## 🎤➡️🤖 Complete Voice-to-AI Pipeline

### Current Workflow:

```
1. 🎤 User speaks into microphone/uploads audio file
   ↓
2. 🔄 Audio gets processed by Whisper-tiny model
   ↓ 
3. 📝 Speech is transcribed to English text
   ↓
4. 🧠 Text is sent to your main model: "model/Whisper-psychology-gemma-3-1b"
   ↓
5. 🔍 FAISS searches relevant documents for context
   ↓
6. 💬 Main model generates psychological response
   ↓
7. 📺 Response is displayed in chat
   ↓
8. 🔊 (Optional) Response can be converted to speech via TTS
```

### Technical Implementation:

#### Step 1-3: Speech-to-Text
```python
# Audio processing with Whisper-tiny
transcribed_text = transcribe_audio(
    audio_bytes, 
    st.session_state.whisper_model,     # whisper-tiny model
    st.session_state.whisper_processor
)
```

#### Step 4-6: AI Processing  
```python
# Main model processing
answer, sources, metadata = process_medical_query(
    transcribed_text,                    # Your speech as text
    st.session_state.faiss_index,       # Document search
    st.session_state.embedding_model,
    st.session_state.optimal_docs,
    st.session_state.model,             # YOUR MAIN MODEL HERE
    st.session_state.tokenizer,         # model/Whisper-psychology-gemma-3-1b
    **generation_params
)
```

#### Step 7-8: Response Display
```python
# Add to chat and optionally convert to speech
st.session_state.messages.append({
    "role": "assistant", 
    "content": answer,      # Response from your main model
    "sources": sources,
    "metadata": metadata
})
```

### Models Used:

1. **Speech-to-Text**: `stt-model/whisper-tiny/`
   - Converts your voice to English text
   - Language: English only (forced)

2. **Main AI Model**: `model/Whisper-psychology-gemma-3-1b/`  ⭐ **YOUR MODEL**
   - Processes the transcribed text
   - Generates psychological responses
   - Uses RAG with FAISS for context

3. **Text-to-Speech**: `tts-model/Kokoro-82M/`
   - Converts AI response back to speech
   - Currently uses placeholder implementation

4. **Document Search**: `faiss_index/`
   - Provides context for better responses

### Usage:

1. **Click the microphone button** 🎤
2. **Speak your mental health question**
3. **Click "🔄 Transcribe Audio"**
4. **Watch the complete pipeline work automatically:**
   - Your speech → Text
   - Text → Your AI model
   - AI response → Chat
   - Optional: Response → Speech

### What happens when you transcribe:

✅ **Immediate automatic processing** - No manual steps needed!
✅ **Your speech text goes directly to your main model**
✅ **Full psychiatric AI response is generated**
✅ **Complete conversation appears in chat**
✅ **Optional TTS for audio response**

The system now automatically sends your transcribed speech to your `model/Whisper-psychology-gemma-3-1b` model and gets a full AI response without any additional steps!
