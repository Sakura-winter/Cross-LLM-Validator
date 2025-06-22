import streamlit as st
import asyncio
from llm_handler import LLMHandler
import time
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="LLM Validator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f8f9fa;
        color: #333;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .answer-section {
        background-color: white;
        color: #333;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    .original-label {
        background-color: #ffd700;
        color: #333;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .revised-label {
        background-color: #90EE90;
        color: #333;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .defended-label {
        background-color: #add8e6;
        color: #333;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.5rem;
        border-radius: 4px;
        border-left: 4px solid #c62828;
    }
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_llm_handler():
    """Initialize and cache the LLM handler"""
    try:
        return LLMHandler()
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
        st.info("Please set your API keys in the .env file. See sidebar for instructions.")
        return None

def display_answers(answers: Dict[str, Any], title: str, label_class: str):
    """Display answers in a formatted way"""
    st.markdown(f"### {title}")
    
    if not answers:
        st.info("No answers available yet.")
        return
    
    for model_name, answer_data in answers.items():
        with st.container():
            st.markdown(f"""
            <div class="model-card">
                <h4>{answer_data.get('display_name', model_name)}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if "error" in answer_data:
                st.markdown(f"""
                <div class="error-message">
                    <strong>Error:</strong> {answer_data['error']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="answer-section">
                    <span class="{label_class}">{label_class.title()}</span>
                    <div style="margin-top: 0.5rem;">
                        {answer_data['content'].replace(chr(10), '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

async def process_validation(question: str, llm_handler: LLMHandler, ui_placeholders: Dict):
    """Runs the entire validation flow and updates the UI via placeholders."""
    results = {}
    
    # Step 1: Get initial answers
    ui_placeholders["progress"].progress(10, "Step 1/3: Getting initial answers...")
    initial_answers = await llm_handler.get_initial_answers(question)

    for model_name, data in initial_answers.items():
        results[model_name] = {**data, "label": "Original", "label_class": "original-label", "display_name": data.get("name")}
    
    st.session_state.results = results

    # Step 2: Compare answers
    ui_placeholders["progress"].progress(40, "Step 2/3: Comparing answers...")
    answers_are_similar = await llm_handler.are_answers_similar(question, initial_answers)

    # Step 3: Conditional re-feedback
    if answers_are_similar:
        ui_placeholders["progress"].progress(100, "Complete: Models are in agreement.")
        st.session_state.final_message = "✅ Models agreed. No re-feedback needed."
    else:
        ui_placeholders["progress"].progress(70, "Step 3/3: Answers differ, getting revised answers...")
        revised_answers = await llm_handler.get_revised_answers(question, initial_answers)
        
        for model_name, data in revised_answers.items():
            if "content" in results[model_name]:
                results[model_name]["original_content"] = results[model_name]["content"]
            results[model_name].update({**data, "label": "Revised", "label_class": "revised-label"})

        ui_placeholders["progress"].progress(100, "Complete: Re-feedback process finished.")
        st.session_state.final_message = "✅ Re-feedback complete. See final answers below."

    st.session_state.results = results
    st.session_state.processing_complete = True

def main():
    st.markdown('<h1 class="main-header">🧠 LLM Validator</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Configuration")
        llm_handler = get_llm_handler()
        if llm_handler is None:
            st.stop()
        
        st.subheader("Active Models")
        st.info("Using native OpenAI and Gemini APIs.")
        models = llm_handler.get_models_list()
        for model in models:
            st.write(f"• {model['name']} ({model['api']})")
        
        st.markdown("---")
        st.markdown("### Workflow:")
        st.markdown("1.  **Initial Answers**: Get responses from OpenAI and Gemini.\n2.  **Compare**: A judge model checks if they agree.\n3.  **Re-feedback**: If they differ, models revise their answers.")
        
        

    col1, col2 = st.columns([2, 1])
    with col1:
        question = st.text_area("Enter your question:", placeholder="e.g., What is the derivative of x²?", height=100)
        
        if st.button("🚀 Validate with LLMs", type="primary", use_container_width=True):
            if not question.strip():
                st.error("Please enter a question.")
            else:
                st.session_state.clear()
                st.session_state.question = question
                st.session_state.processing_complete = False
                
                # Setup UI placeholders
                main_area = st.empty()
                with main_area.container():
                    st.markdown("### Processing...")
                    progress_bar = st.progress(0)
                    
                ui_placeholders = {"progress": progress_bar, "container": main_area}

                try:
                    asyncio.run(process_validation(question, llm_handler, ui_placeholders))
                except Exception as e:
                    st.error(f"An error occurred during validation: {e}")
                
                # Clear placeholders after completion
                main_area.empty()

    with col2:
        st.markdown("### Example Questions:")
        examples = ["What is the derivative of x²?", "If a train travels 60 mph for 2 hours, how far does it go?", "Explain recursion."]
        if 'question' not in st.session_state:
            st.session_state.question = ""
            
        for example in examples:
            if st.button(example, key=f"example_{example[:20]}"):
                st.session_state.question = example
                # Rerunning here is fine to update the text_area
                st.rerun()

    # Display final message if it exists
    if st.session_state.get("final_message"):
        st.success(st.session_state.final_message)

    # Display results if they exist
    if "results" in st.session_state and st.session_state.results:
        st.markdown("---")
        st.markdown("## 📊 Results")
        
        for model_name, answer_data in st.session_state.results.items():
            with st.container():
                st.markdown(f"""<div class="model-card"><h4>{answer_data.get('display_name', model_name)}</h4></div>""", unsafe_allow_html=True)
                if "error" in answer_data:
                    st.markdown(f"""<div class="error-message"><strong>Error:</strong> {answer_data['error']}</div>""", unsafe_allow_html=True)
                else:
                    label = answer_data.get("label", "Answer")
                    label_class = answer_data.get("label_class", "original-label")
                    original_content = answer_data.get("original_content", answer_data.get("content"))
                    if label == "Revised" and answer_data.get("content") == original_content:
                        label = "Defended"
                        label_class = "defended-label"
                    
                    st.markdown(f'<span class="{label_class}">{label}</span>', unsafe_allow_html=True)
                    st.markdown(answer_data['content'])

        st.markdown("---")
        dl_col, clr_col = st.columns(2)
        with dl_col:
            if st.button("📥 Download Results as JSON"):
                import json
                results_to_save = {k: {k2: v2 for k2, v2 in v.items() if k2 not in ['label_class', 'api', 'model_id']} for k, v in st.session_state.results.items()}
                results_package = {"question": st.session_state.question, "final_answers": results_to_save, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
                st.download_button("Download JSON", json.dumps(results_package, indent=2), f"llm_validator_results_{int(time.time())}.json", "application/json")
        with clr_col:
            if st.button("🔄 Clear Results"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main() 
