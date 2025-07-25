# streamlit_app.py
#v1
import streamlit as st
import logging
import torch
import torch._dynamo
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time

# 🔧 COMPLETE TORCH COMPILATION DISABLE for Windows compatibility
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1" 
os.environ["TORCH_DISABLE_DYNAMO"] = "1"
os.environ["TORCH_LOGS"] = ""

# Reset torch state and disable compilation completely
torch._dynamo.reset()
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# Disable compile API if available
try:
    torch._C._set_compile_api_enabled(False)
except:
    pass

# Force eager mode to avoid compilation issues
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# Logging setup
logging.basicConfig(level=logging.INFO)

# 🎨 Streamlit page configuration
st.set_page_config(
    page_title="Whisper AI-Psychiatric",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎯 Custom CSS for dark theme with green accents
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #a6e3a1 0%, #94e2d5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #313244;
        color: #cdd6f4;
    }
    
    .user-message {
        background-color: #1e1e2e;
        border-left: 4px solid #a6e3a1;
    }
    
    .bot-message {
        background-color: #181825;
        border-left: 4px solid #94e2d5;
    }
    
    .source-box {
        background-color: #1e1e2e;
        padding: 0.8rem;
        border-radius: 0.3rem;
        border-left: 3px solid #a6e3a1;
        margin-top: 1rem;
        color: #bac2de;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: #1e1e2e !important;
        color: #cdd6f4 !important;
        border: 1px solid #313244 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #a6e3a1 !important;
        box-shadow: 0 0 0 1px #a6e3a1 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #181825 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #313244 !important;
        color: #cdd6f4 !important;
        border: 1px solid #45475a !important;
        border-radius: 0.5rem !important;
    }
    
    .stButton > button:hover {
        background-color: #a6e3a1 !important;
        color: #11111b !important;
        border-color: #a6e3a1 !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        color: #cdd6f4 !important;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background-color: #1e2d1e !important;
        color: #a6e3a1 !important;
        border-left: 4px solid #a6e3a1 !important;
    }
    
    .stError {
        background-color: #2d1e1e !important;
        color: #f38ba8 !important;
        border-left: 4px solid #f38ba8 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1e1e2e !important;
        color: #cdd6f4 !important;
        border: 1px solid #313244 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #181825 !important;
        border: 1px solid #313244 !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #a6e3a1 !important;
    }
    
    /* Footer styling */
    .footer-style {
        color: #6c7086 !important;
    }
</style>
""", unsafe_allow_html=True)

# 🚀 Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "faiss_loaded" not in st.session_state:
    st.session_state.faiss_loaded = False

# 📊 Sidebar for model status and settings
with st.sidebar:
    st.header("🔧 Model Status")
    
    # Model loading status indicators
    if st.session_state.model_loaded:
        st.success("✅ Whisper AI Model Loaded")
    else:
        st.error("❌ Model Not Loaded")
    
    if st.session_state.faiss_loaded:
        st.success("✅ FAISS Index Loaded")
    else:
        st.error("❌ FAISS Index Not Loaded")
    
    st.divider()
    
    # Enhanced Settings with more options
    st.header("⚙️ Generation Settings")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.6, 0.1, 
                           help="Controls randomness. Lower = more deterministic")
    max_length = st.slider("Max Length", 512, 4096, 2048, 128,
                          help="Maximum total length of generated text")
    top_k = st.slider("Top K", 1, 100, 40, 1,
                     help="Limits sampling to top k tokens")
    top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05,
                     help="Nucleus sampling threshold")
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1, 0.1,
                                     help="Penalty for repeating tokens")
        num_return_sequences = st.slider("Number of Sequences", 1, 3, 1, 1,
                                       help="Number of response variants")
        early_stopping = st.checkbox("Early Stopping", value=True,
                                    help="Stop generation when EOS token is reached")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 🔄 Model Loading Functions
@st.cache_resource
def load_faiss_index():
    """Load FAISS vectorstore with caching and get document count"""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        faiss_index = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        
        # Get the total number of documents in the FAISS index
        total_docs = faiss_index.index.ntotal
        
        # Calculate optimal number of documents to retrieve (20-30% of total, min 3, max 10)
        optimal_docs = max(3, min(10, int(total_docs * 0.25)))
        
        return faiss_index, embedding_model, optimal_docs
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None, None, 3

@st.cache_resource
def load_model():
    """Load the language model with caching"""
    try:
        model_path = "model/Whisper-psychology-gemma-3-1b"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

# 🧠 Enhanced Core Functions with Crisis Detection
def detect_crisis_indicators(question: str) -> tuple:
    """
    Detect crisis indicators in user input
    
    Returns:
        tuple: (is_crisis, crisis_level, crisis_type)
        crisis_level: 'high', 'moderate', 'low', 'none'
        crisis_type: 'suicide', 'self_harm', 'abuse', 'medical_emergency', 'severe_distress', 'none'
    """
    question_lower = question.lower()
    
    # High-risk suicide indicators
    high_suicide_keywords = [
        'kill myself', 'end my life', 'suicide', 'suicidal', 'want to die', 
        'planning to hurt myself', 'planning suicide', 'going to kill myself',
        'better off dead', 'ending it all', 'no point in living'
    ]
    
    # Self-harm indicators
    self_harm_keywords = [
        'cut myself', 'harm myself', 'hurt myself', 'cutting', 'self harm',
        'self-harm', 'burning myself', 'punishing myself'
    ]
    
    # Abuse indicators
    abuse_keywords = [
        'being abused', 'someone is hurting me', 'domestic violence',
        'being threatened', 'unsafe at home', 'afraid for my safety'
    ]
    
    # Severe distress indicators
    severe_distress_keywords = [
        'want to disappear', 'cant take it anymore', "can't go on",
        'hopeless', 'no way out', 'trapped', 'overwhelmed'
    ]
    
    # Check for high-risk crisis
    for keyword in high_suicide_keywords:
        if keyword in question_lower:
            return True, 'high', 'suicide'
    
    for keyword in self_harm_keywords:
        if keyword in question_lower:
            return True, 'high', 'self_harm'
    
    for keyword in abuse_keywords:
        if keyword in question_lower:
            return True, 'high', 'abuse'
    
    # Check for moderate-risk distress
    for keyword in severe_distress_keywords:
        if keyword in question_lower:
            return True, 'moderate', 'severe_distress'
    
    return False, 'none', 'none'

def generate_crisis_response(crisis_level: str, crisis_type: str) -> str:
    """Generate appropriate crisis response based on severity and type"""
    
    emergency_contacts = """
🚨 **IMMEDIATE HELP AVAILABLE:**
• **Sri Lanka Crisis Helpline**: 1926 (24/7)
• **Emergency Services**: 119
• **Samaritans of Sri Lanka**: 071-5-1426-26
• **SLACPH Hotline**: 091-2-34-99-34

**International:**
• **Crisis Text Line**: Text HOME to 741741
• **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/
"""
    
    if crisis_level == 'high':
        if crisis_type == 'suicide':
            return f"""🚨 **CRISIS ALERT: IMMEDIATE ACTION REQUIRED** 🚨

I'm very concerned about your safety right now. Your life has value and there are people who want to help.

{emergency_contacts}

**IMMEDIATE STEPS:**
1. **Call 1926 or 119 right now** - Don't wait
2. **Go to the nearest hospital emergency room**
3. **Reach out to a trusted friend, family member, or counselor immediately**
4. **Remove any means of harm from your immediate environment**

**Remember:** 
- This feeling is temporary, even though it doesn't feel that way right now
- You are not alone - help is available 24/7
- Your life matters and things can get better with proper support

I'm also alerting available support specialists about this conversation. Please prioritize your safety above all else right now."""

        elif crisis_type == 'self_harm':
            return f"""🚨 **URGENT: SELF-HARM CRISIS DETECTED** 🚨

I'm deeply concerned about your safety. Self-harm might feel like a way to cope, but there are safer alternatives and people who can help.

{emergency_contacts}

**IMMEDIATE STEPS:**
1. **Call 1926 for immediate support**
2. **Remove or distance yourself from any items you might use to self-harm**
3. **Contact a trusted person - friend, family, counselor**
4. **Consider going to an emergency room if urges are strong**

**Safer coping alternatives:**
- Hold ice cubes in your hands
- Draw on your skin with a red marker
- Exercise intensely for a few minutes
- Call or text someone you trust

You deserve care and healing, not pain. Please reach out for help right now."""

        elif crisis_type == 'abuse':
            return f"""🚨 **SAFETY ALERT: ABUSE SITUATION DETECTED** 🚨

Your safety is the top priority. If you're in immediate danger, please take action to protect yourself.

**IMMEDIATE SAFETY RESOURCES:**
• **Emergency Services**: 119
• **Women & Children's Bureau**: 011-2-43-34-44
• **Women in Crisis Hotline**: 077-6-9-64-7-64
• **Police Emergency**: 119

{emergency_contacts}

**SAFETY STEPS:**
1. **If in immediate danger, call 119**
2. **Get to a safe location if possible**
3. **Contact trusted friends or family who can help**
4. **Document any injuries or incidents safely**
5. **Consider contacting local authorities or support organizations**

**Remember:** 
- The abuse is not your fault
- You deserve to be safe and treated with respect
- Help and safe housing options are available

Please prioritize your physical safety. Support specialists are being alerted to provide additional assistance."""

    elif crisis_level == 'moderate':
        return f"""💙 **SUPPORT ALERT: EMOTIONAL DISTRESS DETECTED** 💙

I can hear that you're going through a really difficult time right now. Your feelings are valid, and you don't have to face this alone.

**SUPPORT RESOURCES:**
{emergency_contacts}

**IMMEDIATE SUPPORT STEPS:**
1. **Consider calling 1926** - they're trained to help with exactly what you're experiencing
2. **Reach out to someone you trust** - a friend, family member, counselor, or spiritual leader
3. **Focus on your immediate safety and basic needs** - food, water, rest
4. **Try grounding techniques** - name 5 things you can see, 4 things you can touch, 3 things you can hear

**Remember:**
- These intense feelings, while overwhelming, are temporary
- You've survived difficult times before
- Small steps toward getting help can make a big difference
- You are worthy of care and support

Would you like to talk about some specific coping strategies, or would you prefer information about professional support options?"""

    return ""

def generate_prompt(context: str, question: str) -> str:
    """Generate enhanced prompt for the model with better structure and crisis awareness"""
    
    # Check for crisis indicators first
    is_crisis, crisis_level, crisis_type = detect_crisis_indicators(question)
    
    if is_crisis and crisis_level == 'high':
        # For high-risk situations, return crisis response immediately
        return generate_crisis_response(crisis_level, crisis_type)
    
    # For non-crisis or moderate crisis, generate enhanced prompt with safety guidelines
    safety_guidelines = """
CRITICAL SAFETY PROTOCOLS:
- ALWAYS prioritize user safety and well-being above all else
- If user mentions suicide, self-harm, or abuse, provide immediate crisis resources
- Never provide harmful advice or suggestions
- Always validate feelings while promoting safety
- Encourage professional help for serious mental health concerns
- Use trauma-informed, culturally sensitive language
- If uncertain about user safety, err on the side of caution and provide crisis resources

CRISIS RESOURCES TO INCLUDE WHEN APPROPRIATE:
- Sri Lanka Crisis Helpline: 1926 (24/7)
- Emergency Services: 119
- Samaritans of Sri Lanka: 071-5-1426-26
"""
    
    crisis_notice = ""
    if is_crisis and crisis_level == 'moderate':
        crisis_notice = f"\n🚨 MODERATE CRISIS DETECTED: {crisis_type.upper()} - Include appropriate support resources in response.\n"
    
    return f"""<|im_start|>system
You are Whisper, a compassionate mental health assistant developed by DeepFinders at SLTC Research University. You offer accurate, supportive psychological guidance based on the given context. Always be empathetic, professional, and communicate with clarity and care.

{safety_guidelines}

Guidelines:
- Provide comprehensive, detailed responses when needed to fully address the user's concerns
- Use evidence-based psychological principles and therapeutic approaches
- Break down complex concepts into understandable terms
- Offer practical coping strategies and actionable advice
- Be empathetic and validate the user's feelings
- If uncertain about specific medical advice, acknowledge limitations and suggest professional consultation
- Focus on promoting mental wellness and resilience
- Consider cultural sensitivity and individual differences
- Provide resources or techniques that can be helpful for mental health
- ALWAYS assess for safety concerns and provide crisis resources when needed

{crisis_notice}
<|im_end|>

<|im_start|>context
Comprehensive Medical and Psychological Context:
{context}
<|im_end|>

<|im_start|>human
{question}
<|im_end|>

<|im_start|>assistant
"""

def generate_response(prompt: str, model, tokenizer, **kwargs) -> str:
    """
    Enhanced response generation function based on the reference code
    
    Args:
        prompt (str): Input prompt
        model: The language model
        tokenizer: The tokenizer
        **kwargs: Generation parameters
    
    Returns:
        str: Generated response
    """
    try:
        # Extract parameters with defaults
        max_length = kwargs.get('max_length', 512)
        temperature = kwargs.get('temperature', 0.7)
        top_k = kwargs.get('top_k', 50)
        top_p = kwargs.get('top_p', 0.95)
        repetition_penalty = kwargs.get('repetition_penalty', 1.1)
        num_return_sequences = kwargs.get('num_return_sequences', 1)
        early_stopping = kwargs.get('early_stopping', True)
        
        # Device selection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Tokenize the input prompt
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096,  # Increased input length for more context
            padding=True
        ).to(device)
        
        # Calculate max_new_tokens to avoid exceeding model limits
        input_length = inputs['input_ids'].shape[1]
        max_new_tokens = min(max_length - input_length, 2048)  # Allow longer responses
        
        if max_new_tokens <= 50:  # Lowered threshold for minimum response length
            return "Input too long. Please try a shorter question."
        
        # Generate output with enhanced parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,  # Maximum total length
                max_new_tokens=max_new_tokens,  # Maximum new tokens to generate
                temperature=temperature,  # Controls randomness
                top_k=top_k,  # Limits sampling to top k tokens
                top_p=top_p,  # Nucleus sampling
                do_sample=True,  # Enable sampling for varied outputs
                num_return_sequences=num_return_sequences,  # Number of sequences
                repetition_penalty=repetition_penalty,  # Prevent repetition
                early_stopping=early_stopping,  # Stop at EOS token
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                no_repeat_ngram_size=2,  # Prevent repeating 2-grams
                length_penalty=1.0,  # Neutral length penalty
            )
        
        # Handle multiple sequences
        responses = []
        for i in range(num_return_sequences):
            # Extract only the new tokens (excluding the input prompt)
            new_tokens = outputs[i][input_length:]
            
            # Decode the output to text
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up the response
            generated_text = generated_text.strip()
            
            # Remove any incomplete sentences at the end
            if generated_text and not generated_text.endswith(('.', '!', '?', ':')):
                # Find the last complete sentence
                last_punct = max(
                    generated_text.rfind('.'),
                    generated_text.rfind('!'),
                    generated_text.rfind('?'),
                    generated_text.rfind(':')
                )
                if last_punct > len(generated_text) * 0.5:  # Only truncate if we keep most of the text
                    generated_text = generated_text[:last_punct + 1]
            
            responses.append(generated_text)
        
        # Return the best response (or combine if multiple)
        if num_return_sequences == 1:
            return responses[0] if responses[0] else "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        else:
            # For multiple sequences, return the longest meaningful response
            best_response = max(responses, key=len) if responses else ""
            return best_response if best_response else "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
    
    except Exception as e:
        error_msg = f"Sorry, I encountered an error generating a response: {str(e)}"
        logging.error(f"Generation error: {e}")
        return error_msg

def process_medical_query(question: str, faiss_index, embedding_model, optimal_docs, model, tokenizer, **generation_params) -> tuple:
    """
    Process a medical query end-to-end with crisis detection
    
    Args:
        question: User's question
        faiss_index: FAISS vector store
        embedding_model: Embedding model
        optimal_docs: Optimal number of documents to retrieve
        model: Language model
        tokenizer: Tokenizer
        **generation_params: Generation parameters
    
    Returns:
        tuple: (response, sources, metadata)
    """
    try:
        # First, check for crisis indicators
        is_crisis, crisis_level, crisis_type = detect_crisis_indicators(question)
        
        if is_crisis and crisis_level == 'high':
            # For high-risk situations, return crisis response immediately without model generation
            crisis_response = generate_crisis_response(crisis_level, crisis_type)
            metadata = {
                "crisis_detected": True,
                "crisis_level": crisis_level,
                "crisis_type": crisis_type,
                "escalated": True,
                "message": "Emergency response activated - professional support recommended"
            }
            return crisis_response, [], metadata
        
        # Retrieve relevant documents using auto-calculated number
        docs = faiss_index.similarity_search(question, k=optimal_docs)
        
        # Prepare context
        context_parts = []
        sources = []
        
        for i, doc in enumerate(docs, 1):
            # Add document content to context
            context_parts.append(f"Document {i}:\n{doc.page_content}")
            
            # Prepare source information
            sources.append({
                "source": doc.metadata.get('source', f'Document {i}'),
                "content": doc.page_content[:300],  # First 300 chars for preview
                "relevance_score": getattr(doc, 'relevance_score', 'N/A')
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate enhanced prompt
        prompt = generate_prompt(context, question)
        
        # For moderate crisis, check if prompt returned crisis response directly
        if prompt.startswith("💙 **SUPPORT ALERT"):
            metadata = {
                "crisis_detected": True,
                "crisis_level": crisis_level,
                "crisis_type": crisis_type,
                "escalated": False,
                "message": "Moderate crisis detected - support resources provided"
            }
            return prompt, sources, metadata
        
        # Generate response using the model
        response = generate_response(prompt, model, tokenizer, **generation_params)
        
        # Metadata
        metadata = {
            "num_sources": len(sources),
            "context_length": len(context),
            "prompt_length": len(prompt),
            "optimal_docs_used": optimal_docs,
            "generation_params": generation_params,
            "crisis_detected": is_crisis,
            "crisis_level": crisis_level if is_crisis else "none",
            "crisis_type": crisis_type if is_crisis else "none"
        }
        
        return response, sources, metadata
        
    except Exception as e:
        error_response = f"I apologize, but I encountered an error processing your question: {str(e)}"
        return error_response, [], {"error": str(e)}

# 🚀 Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Whisper AI-Psychiatric</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #94e2d5; font-size: 1.2rem; margin-bottom: 2rem;">Developed by DeepFinders</p>', unsafe_allow_html=True)
    
    # Emergency contacts section
    with st.expander("🚨 Emergency Mental Health Resources", expanded=False):
        st.markdown("""
        ### 🇱🇰 Sri Lanka Emergency Contacts
        - **National Crisis Helpline**: 1926 (24/7)
        - **Emergency Services**: 119
        - **Samaritans of Sri Lanka**: 071-5-1426-26
        - **SLACPH Mental Health Hotline**: 091-2-34-99-34
        - **Mental Health Foundation**: 011-2-68-9909
        
        ### 🌍 International Resources
        - **Crisis Text Line**: Text HOME to 741741
        - **International Association for Suicide Prevention**: [IASP Crisis Centers](https://www.iasp.info/resources/Crisis_Centres/)
        - **Befrienders Worldwide**: [Find local support](https://www.befrienders.org/)
        
        ### ⚠️ When to Seek Immediate Help
        - Thoughts of suicide or self-harm
        - Feeling unsafe or in danger
        - Severe emotional distress
        - Substance abuse crisis
        - Any mental health emergency
        
        **Remember: You are not alone. Help is always available.**
        """)
    
    st.markdown("---")
    
    # Load models with progress indication
    if not st.session_state.model_loaded or not st.session_state.faiss_loaded:
        with st.spinner("Loading models... This may take a few minutes on first run."):
            # Load FAISS index
            if not st.session_state.faiss_loaded:
                faiss_index, embedding_model, optimal_docs = load_faiss_index()
                if faiss_index is not None:
                    st.session_state.faiss_index = faiss_index
                    st.session_state.embedding_model = embedding_model
                    st.session_state.optimal_docs = optimal_docs
                    st.session_state.faiss_loaded = True
            
            # Load language model
            if not st.session_state.model_loaded:
                model, tokenizer = load_model()
                if model is not None:
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model_loaded = True
        
        if st.session_state.model_loaded and st.session_state.faiss_loaded:
            st.success("🎉 All models loaded successfully!")
            time.sleep(1)  # Brief pause for user to see success message
            st.rerun()
        else:
            st.error("❌ Failed to load models. Please check your model and index paths.")
            return
    
    # Chat interface
    st.subheader("💬 Chat with Whisper AI-Psychiatric")
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.container():
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.container():
                # Check if this is a crisis response
                is_crisis_response = False
                crisis_level = "none"
                if "metadata" in message:
                    is_crisis_response = message["metadata"].get("crisis_detected", False)
                    crisis_level = message["metadata"].get("crisis_level", "none")
                
                # Display crisis alert if applicable
                if is_crisis_response and crisis_level == "high":
                    st.error("🚨 CRISIS ALERT: Emergency resources have been provided. Please prioritize your immediate safety.")
                elif is_crisis_response and crisis_level == "moderate":
                    st.warning("💙 SUPPORT ALERT: Support resources have been provided. Your well-being is important.")
                
                # Clean the message content to remove any HTML tags
                clean_content = message["content"].replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Whisper:</strong><br>
                    {clean_content}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 View Sources & Details"):
                        # Show metadata if available
                        if "metadata" in message:
                            metadata = message["metadata"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Sources Used", metadata.get("num_sources", "N/A"))
                            with col2:
                                st.metric("Context Length", metadata.get("context_length", "N/A"))
                            with col3:
                                st.metric("Prompt Length", metadata.get("prompt_length", "N/A"))
                            st.divider()
                        
                        # Show sources
                        for i, source in enumerate(message["sources"], 1):
                            source_content = source.get('content', '')[:200].replace("<", "&lt;").replace(">", "&gt;")
                            source_name = source.get('source', 'Unknown').replace("<", "&lt;").replace(">", "&gt;")
                            relevance = source.get('relevance_score', 'N/A')
                            
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {i}:</strong> {source_name}<br>
                                <strong>Relevance:</strong> {relevance}<br>
                                <em>Content preview:</em> {source_content}...
                            </div>
                            """, unsafe_allow_html=True)
    
    # Chat input
    user_question = st.chat_input("Ask me anything about mental health topics...")
    
    if user_question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Generate response with enhanced parameters
        with st.spinner("Thinking... 🤔"):
            generation_params = {
                'max_length': max_length,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'repetition_penalty': repetition_penalty,
                'num_return_sequences': num_return_sequences,
                'early_stopping': early_stopping
            }
            
            # Process the query
            answer, sources, metadata = process_medical_query(
                user_question,
                st.session_state.faiss_index,
                st.session_state.embedding_model,
                st.session_state.optimal_docs,
                st.session_state.model,
                st.session_state.tokenizer,
                **generation_params
            )
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources,
                "metadata": metadata
            })
        
        # Rerun to display the new messages
        st.rerun()

    # 📱 Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        🧠 Whisper AI-Psychiatric - Developed by DeepFinders<br>
        Powered by HuggingFace Transformers & LangChain | Enhanced Generation Pipeline
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()