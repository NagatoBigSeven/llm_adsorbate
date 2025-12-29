import sys
import os
import re
import tempfile
import uuid

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from src.agent.agent import get_agent_executor, _prepare_initial_state
from src.utils.config import get_api_key, save_api_key, is_env_key_set

st.set_page_config(page_title="LLM Agent Demo", layout="wide")
st.title("adsKRK")

@st.cache_resource
def initialize_agent_executor():
    return get_agent_executor()

agent_executor = initialize_agent_executor()

def render_message(content):
    parts = re.split(r"(```python\n.*\n```)", content, flags=re.DOTALL)
    for part in parts:
        if part.strip():
            if part.startswith("```python"):
                code_to_display = part.split("\n", 1)[1].rsplit("\n```", 1)[0]
                st.code(code_to_display, language="python")
            else:
                st.markdown(part)

def render_message_in_status(content, status):
    parts = re.split(r"(```python\n.*\n```)", content, flags=re.DOTALL)
    for part in parts:
        if part.strip():
            if part.startswith("```python"):
                code_to_display = part.split("\n", 1)[1].rsplit("\n```", 1)[0]
                status.code(code_to_display, language="python")
            else:
                status.markdown(part)

# --- API Key Configuration ---
st.sidebar.header("Settings")

# Load API key from available sources
saved_key, key_source = get_api_key()
env_key_active = is_env_key_set()

# Show source indicator
if key_source == "env":
    st.sidebar.success("🌍 API Key loaded from environment variable")
elif key_source == "config":
    st.sidebar.info("📁 API Key loaded from config file")
else:
    st.sidebar.caption("✏️ Enter your OpenRouter API Key")

# Show warning if env var is set
if env_key_active:
    st.sidebar.warning("⚠️ Environment variable is active. Saving to config will NOT override it.")

# API key input field
openrouter_api_key = st.sidebar.text_input(
    "OpenRouter API Key", 
    value=saved_key or "",
    type="password", 
    key="openrouter_api_key"
)

# Save checkbox (disabled if env var is active)
save_key_checkbox = st.sidebar.checkbox(
    "Save API Key for future sessions",
    disabled=env_key_active,
    help="Saves to ~/.adskrk/config.json" if not env_key_active else "Cannot save while environment variable is active"
)

# Note: API key is now passed directly to agent state, not via global env var
# This prevents key leakage between concurrent sessions
if openrouter_api_key:
    # Save if checkbox is checked and key is different from saved
    if save_key_checkbox and openrouter_api_key != saved_key:
        if save_api_key(openrouter_api_key):
            st.sidebar.success("✅ API Key saved!")


st.sidebar.header("Inputs")
smiles_input = st.sidebar.text_input("SMILES String")
xyz_file = st.sidebar.file_uploader("Slab XYZ file", type=['xyz'])
user_query = st.sidebar.text_area("User Query", value="")

run_button = st.sidebar.button("Run Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_message(message["content"])

if run_button:
    if not openrouter_api_key:
        st.sidebar.error("Please enter your OpenRouter API Key.")
    elif not smiles_input:
        st.sidebar.error("Please enter a SMILES string.")
    elif not xyz_file:
        st.sidebar.error("Please upload a slab XYZ file.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xyz", mode='w') as tmp_file:
            xyz_content = xyz_file.getvalue().decode('utf-8')
            tmp_file.write(xyz_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Generate unique session ID for file isolation
            session_id = str(uuid.uuid4())[:8]  # First 8 chars for brevity
            
            initial_state = _prepare_initial_state(
                smiles=smiles_input, 
                slab_path=tmp_file_path, 
                user_request=user_query,
                api_key=openrouter_api_key,
                session_id=session_id
            )
            
            st.session_state.messages.append({"role": "user", "content": f"**Inputs provided:**\n- SMILES: `{smiles_input}`\n- Slab file: `{xyz_file.name}`\n- Query: `{user_query}`\n\n**Generated prompt for the agent...**"})
            with st.chat_message("user"):
                st.markdown(f"**Inputs provided:**\n- SMILES: `{smiles_input}`\n- Slab file: `{xyz_file.name}`\n- Query: `{user_query}`")

            with st.chat_message("assistant"):
                final_answer = ""
                with st.status("Thinking...", expanded=True) as status:
                    MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "20"))
                    step_count = 0
                    recent_messages = []
                    recent_tool_calls = []

                    for event in agent_executor.stream(
                        initial_state,
                        stream_mode="values",
                    ):
                        step_count += 1
                        if step_count >= MAX_STEPS:
                            status.markdown("⚠️ **WARNING: Reached maximum step limit. Terminating to prevent infinite loop.**")
                            break
                        if "tool_calls" in event:
                            for tc in event["tool_calls"]:
                                tool_sig = f"{tc['name']}:{str(tc['args'])}"
                                # Check for repeated tool calls
                                if recent_tool_calls.count(tool_sig) >= 3:
                                    status.markdown("⚠️ **WARNING: Detected repeated tool calls. Possible loop.**")
                                recent_tool_calls.append(tool_sig)
                                if len(recent_tool_calls) > 10:
                                    recent_tool_calls.pop(0)
                                status.markdown(f"Calling tool: `{tc['name']}` with args: `{tc['args']}`")
                            status.divider()
                        if "tool_output" in event:
                            for to in event["tool_output"]:
                                status.markdown(f"Tool output: `{to}`")
                            status.divider()
                        if "messages" in event:
                            last_message = event["messages"][-1]
                            if last_message.type == "ai" and last_message.content:
                                content = last_message.content
                                if content in recent_messages:
                                    status.markdown("⚠️ **WARNING: Detected repeated message content.**")
                                recent_messages.append(content)
                                if len(recent_messages) > 5:
                                    recent_messages.pop(0)
                                render_message_in_status(content, status)
                                status.divider()
                                final_answer = content
                    
                    status.update(label="Agent finished.", state="complete", expanded=False)

                if final_answer:
                    render_message(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                else:
                    st.warning("The agent did not produce a final answer.")
            
            os.remove(tmp_file_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

st.sidebar.markdown("---")
st.sidebar.info("Provide SMILES, a slab file, and a query, then click 'Run Agent'.")