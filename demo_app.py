"""
LoRA Fine-Tuned Models Demo
Qwen2.5-Coder-1.5B with DEEP and DIVERSE datasets
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Global variables for models
models = {}
tokenizer = None
base_model = None

def load_models():
    """Load base model and both LoRA adapters"""
    global models, tokenizer, base_model
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    
    print("Loading DEEP model...")
    models["DEEP"] = PeftModel.from_pretrained(
        base_model,
        "B0DH1i/qwen-coder-lora-deep"  # Model yüklendikten sonra güncelle
    )
    
    print("Loading DIVERSE model...")
    models["DIVERSE"] = PeftModel.from_pretrained(
        base_model,
        "B0DH1i/qwen-coder-lora-diverse"  # Model yüklendikten sonra güncelle
    )
    
    print("✓ All models loaded!")

def generate_code(
    problem,
    model_choice,
    temperature,
    top_p,
    max_tokens,
    do_sample
):
    """Generate code solution using selected model"""
    
    if not models:
        return "❌ Models not loaded yet. Please wait..."
    
    # System prompt
    system_prompt = "You are an expert Python programmer. Please read the problem carefully before writing any Python code."
    
    # Full prompt
    prompt = f"{system_prompt}\n\nProblem:\n{problem}\n\nSolution:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(models[model_choice].device)
    
    # Generate
    with torch.no_grad():
        outputs = models[model_choice].generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract solution
    solution = generated.split("Solution:\n")[-1].strip()
    
    return solution

def compare_models(
    problem,
    temperature,
    top_p,
    max_tokens,
    do_sample
):
    """Compare both models side by side"""
    
    deep_solution = generate_code(problem, "DEEP", temperature, top_p, max_tokens, do_sample)
    diverse_solution = generate_code(problem, "DIVERSE", temperature, top_p, max_tokens, do_sample)
    
    return deep_solution, diverse_solution

# Example problems
examples = [
    ["Write a Python function that returns the nth Fibonacci number."],
    ["Write a function to check if a string is a palindrome."],
    ["Given an array of integers nums and an integer target, return indices of the two numbers that add up to target."],
    ["Write a function to reverse a linked list."],
    ["Implement a binary search algorithm."]
]

# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.model-badge {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 10px 20px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
    margin: 10px 0;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="LoRA Code Generator") as demo:
    
    gr.Markdown("""
    # 🚀 LoRA Fine-Tuned Code Generator
    
    Compare two fine-tuned versions of **Qwen2.5-Coder-1.5B**:
    - **DEEP Model**: Trained on CodeGen-Deep-5K (deeper reasoning)
    - **DIVERSE Model**: Trained on CodeGen-Diverse-5K (diverse problems)
    
    ---
    """)
    
    with gr.Tabs():
        
        # Tab 1: Single Model
        with gr.Tab("🎯 Single Model"):
            with gr.Row():
                with gr.Column(scale=2):
                    problem_input = gr.Textbox(
                        label="📝 Problem Description",
                        placeholder="Describe your coding problem here...",
                        lines=5
                    )
                    
                    model_choice = gr.Radio(
                        choices=["DEEP", "DIVERSE"],
                        value="DEEP",
                        label="🤖 Select Model",
                        info="Choose which fine-tuned model to use"
                    )
                    
                    with gr.Accordion("⚙️ Generation Settings", open=False):
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="🌡️ Temperature",
                            info="Higher = more creative, Lower = more focused"
                        )
                        
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.95,
                            step=0.05,
                            label="🎲 Top P (Nucleus Sampling)",
                            info="Probability threshold for token selection"
                        )
                        
                        max_tokens = gr.Slider(
                            minimum=128,
                            maximum=1024,
                            value=512,
                            step=64,
                            label="📏 Max Tokens",
                            info="Maximum length of generated code"
                        )
                        
                        do_sample = gr.Checkbox(
                            value=True,
                            label="🎰 Enable Sampling",
                            info="Use sampling for generation (recommended)"
                        )
                    
                    generate_btn = gr.Button("✨ Generate Code", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    output_code = gr.Code(
                        label="💻 Generated Solution",
                        language="python",
                        lines=20
                    )
            
            gr.Examples(
                examples=examples,
                inputs=[problem_input],
                label="📚 Example Problems"
            )
            
            generate_btn.click(
                fn=generate_code,
                inputs=[problem_input, model_choice, temperature, top_p, max_tokens, do_sample],
                outputs=[output_code]
            )
        
        # Tab 2: Compare Models
        with gr.Tab("⚖️ Compare Models"):
            gr.Markdown("""
            ### Compare both models side-by-side
            See how DEEP and DIVERSE models solve the same problem differently.
            """)
            
            with gr.Row():
                with gr.Column():
                    problem_compare = gr.Textbox(
                        label="📝 Problem Description",
                        placeholder="Describe your coding problem here...",
                        lines=5
                    )
                    
                    with gr.Accordion("⚙️ Generation Settings", open=False):
                        temp_compare = gr.Slider(0.1, 2.0, 0.7, 0.1, label="🌡️ Temperature")
                        top_p_compare = gr.Slider(0.1, 1.0, 0.95, 0.05, label="🎲 Top P")
                        max_tokens_compare = gr.Slider(128, 1024, 512, 64, label="📏 Max Tokens")
                        sample_compare = gr.Checkbox(True, label="🎰 Enable Sampling")
                    
                    compare_btn = gr.Button("⚖️ Compare Both Models", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="model-badge">🔵 DEEP Model</div>')
                    deep_output = gr.Code(label="DEEP Solution", language="python", lines=15)
                
                with gr.Column():
                    gr.HTML('<div class="model-badge">🟢 DIVERSE Model</div>')
                    diverse_output = gr.Code(label="DIVERSE Solution", language="python", lines=15)
            
            gr.Examples(
                examples=examples,
                inputs=[problem_compare],
                label="📚 Example Problems"
            )
            
            compare_btn.click(
                fn=compare_models,
                inputs=[problem_compare, temp_compare, top_p_compare, max_tokens_compare, sample_compare],
                outputs=[deep_output, diverse_output]
            )
        
        # Tab 3: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## 📖 About This Demo
            
            ### Models
            - **Base Model**: Qwen2.5-Coder-1.5B-Instruct
            - **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
            - **Training**: 8-bit quantization, 512 token context
            
            ### Datasets
            - **DEEP**: CodeGen-Deep-5K - Focus on deeper reasoning traces
            - **DIVERSE**: CodeGen-Diverse-5K - Focus on diverse problem types
            
            ### Training Details
            - **LoRA Rank**: 32
            - **LoRA Alpha**: 64
            - **Learning Rate**: 2e-4
            - **Batch Size**: 32 (effective)
            - **Context Length**: 512 tokens
            
            ### Parameters Explained
            
            #### 🌡️ Temperature
            - **Low (0.1-0.5)**: More deterministic, focused code
            - **Medium (0.6-0.9)**: Balanced creativity and accuracy
            - **High (1.0-2.0)**: More creative, diverse solutions
            
            #### 🎲 Top P (Nucleus Sampling)
            - Controls diversity by limiting token selection
            - **0.9-0.95**: Recommended for code generation
            - **Lower**: More focused, **Higher**: More diverse
            
            #### 📏 Max Tokens
            - Maximum length of generated code
            - **128-256**: Short functions
            - **512**: Medium complexity
            - **1024**: Complex implementations
            
            #### 🎰 Sampling
            - **Enabled**: Uses temperature and top_p (recommended)
            - **Disabled**: Greedy decoding (deterministic)
            
            ### Links
            - 🔗 [GitHub Repository](https://github.com/B0DH1i/Lora-fine-tune)
            - 🤗 [DEEP Model on HuggingFace](#)
            - 🤗 [DIVERSE Model on HuggingFace](#)
            
            ---
            
            **Made with ❤️ using Gradio and HuggingFace Transformers**
            """)

# Load models on startup
print("Initializing models...")
try:
    load_models()
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    print("Models will be loaded on first use.")

# Launch
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )
