from groq import Groq
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from crewai import Crew, Agent, Task, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
import json
import os
import requests
from langchain.tools import tool
from langchain.agents import load_tools
from langchain_openai import ChatOpenAI
from crewai_tools import tool
from crewai import Crew, Process
import tomllib
from langchain_groq import ChatGroq
import pandas as pd
from datetime import datetime

# Add this near the top of the file, after the imports
if 'download_content' not in st.session_state:
    st.session_state.download_content = None

# Set page config
st.set_page_config(
    page_title="Autonomous Crew Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .section-header {
        color: #1E3D59;
        padding: 1rem 0;
        border-bottom: 2px solid #1E3D59;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for API key and general info
with st.sidebar:
    st.title("⚙️ Settings")
    groq_api_key = st.text_input('Enter your GROQ API key', type='password')
    st.markdown("---")
    st.markdown("""
        ### How to use this app:
        1. Enter your GROQ API key
        2. Provide the user input
        3. Go to the Configure Agents tab
        4. Define the number of agents
        5. Configure each agent's details
        6. Click 'Create Crew' to start
        7. Go to the Download tab to download configuration and results
    """)

# Main content
st.title('🤖 Autonomous Crew Builder')
st.markdown("""
    Create an autonomous crew of AI agents that work together to achieve your goals. 
    Each agent can be assigned specific roles, goals, and tasks.
""")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["User Input", "Configure Agents", "Download"])

# Create a container for the results (moved to the top)
results_container = st.container()

with tab1:
    st.markdown('<h3 class="section-header">User Input</h3>', unsafe_allow_html=True)
    human_input = st.text_area(
        "User Input",
        help="Enter any additional information or context that the agents should consider when executing their tasks",
        height=150
    )

with tab2:
    st.markdown('<h3 class="section-header">Agent Configuration</h3>', unsafe_allow_html=True)
    number_of_agents = st.number_input(
        'Number of Agents',
        min_value=1,
        max_value=10,
        value=1,
        help="Select how many agents you want in your crew"
    )

    # Create a container for agent configurations
    agent_configs = []
    for i in range(number_of_agents):
        with st.expander(f"Agent {i+1} Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                agent_name = st.text_input(f"Name", key=f"name_{i}")
                role = st.text_input(f"Role", key=f"role_{i}")
                goal = st.text_input(f"Goal", key=f"goal_{i}")
            with col2:
                backstory = st.text_area(f"Backstory", key=f"backstory_{i}")
                task_description = st.text_area(f"Task Description", key=f"task_{i}")
                expected_output = st.text_area(f"Expected Output", key=f"output_{i}")
            
            agent_configs.append({
                "name": agent_name,
                "role": role,
                "goal": goal,
                "backstory": backstory,
                "task": task_description,
                "output": expected_output
            })

    # Create Crew button (moved inside tab2)
    if st.button('🚀 Create Crew', type="primary"):
        if not groq_api_key:
            st.error("Please enter your GROQ API key in the sidebar!")
        else:
            with st.spinner("Creating and running your crew..."):
                os.environ["GROQ_API_KEY"] = groq_api_key
                client = Groq()

                GROQ_LLM = ChatGroq(
                    model="llama-3.1-8b-instant"
                )

                agentlist = []
                tasklist = []
                
                for config in agent_configs:
                    agent = Agent(
                        role=config["role"],
                        goal=config["goal"],
                        backstory=config["backstory"],
                        llm=GROQ_LLM,
                        verbose=True,
                        allow_delegation=False,
                        max_iter=5,
                        memory=True
                    )
                    agentlist.append(agent)
                    
                    task = Task(
                        description=config["task"] + "\n\nAdditional Context: " + human_input,
                        expected_output=config["output"],
                        agent=agent
                    )
                    tasklist.append(task)

                crew = Crew(
                    agents=agentlist,
                    tasks=tasklist,
                    process=Process.sequential,
                    full_output=True,
                    share_crew=False,
                )

                # Kick off the crew's work
                results = crew.kickoff()

                # Display results in an organized way
                with results_container:
                    st.markdown('<h3 class="section-header">Results</h3>', unsafe_allow_html=True)
                    
                    # Create tabs for different views (removed Download Report tab)
                    results_tab1, results_tab2 = st.tabs(["Detailed Output", "Summary"])
                    
                    with results_tab1:
                        for i, task in enumerate(tasklist):
                            with st.expander(f"Agent {i+1}: {agent_configs[i]['name']}", expanded=True):
                                st.markdown("**Task:**")
                                st.write(task.description)
                                st.markdown("**Output:**")
                                st.write(task.output.exported_output)
                    
                    with results_tab2:
                        # Create a summary DataFrame
                        summary_data = []
                        for i, task in enumerate(tasklist):
                            summary_data.append({
                                "Agent": f"{i+1}: {agent_configs[i]['name']}",
                                "Role": agent_configs[i]['role'],
                                "Output": task.output.exported_output
                            })
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df)

                    # Move download button to tab3
                    with tab3:
                        st.markdown('<h3 class="section-header">Download Report</h3>', unsafe_allow_html=True)
                        if st.session_state.download_content is None:
                            st.markdown("""
                            After creating and running your crew, you can download the complete report here.
                            
                            The report will include:
                            - Complete configuration of all agents
                            - User input and context
                            - Detailed results from each agent
                            - Timestamps for configuration and execution
                            
                            Click 'Create Crew' in the Configure Agents tab to generate results, then return here to download them.
                            """)
                        else:
                            st.markdown("""
                            Your crew has been created and the results are ready for download.
                            
                            The report includes:
                            - Complete configuration of all agents
                            - User input and context
                            - Detailed results from each agent
                            - Timestamps for configuration and execution
                            """)
                            
                            st.download_button(
                                label="📥 Download Complete Report",
                                data=st.session_state.download_content,
                                file_name=f"crew_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )

                    # Replace the existing download section in the results with this:
                    combined_text = f"""AUTONOMOUS CREW BUILDER - COMPLETE REPORT
{'='*50}

PART 1: CREW CONFIGURATION
{'='*50}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Additional Context:
{human_input}

Agent Configurations:
"""
                    for i, config in enumerate(agent_configs):
                        combined_text += f"\nAgent {i+1}: {config['name']}\n"
                        combined_text += f"Role: {config['role']}\n"
                        combined_text += f"Goal: {config['goal']}\n"
                        combined_text += f"Backstory: {config['backstory']}\n"
                        combined_text += f"Task: {config['task']}\n"
                        combined_text += f"Expected Output: {config['output']}\n"
                        combined_text += "-" * 50 + "\n"

                    combined_text += f"""
{'='*50}
PART 2: CREW RESULTS
{'='*50}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Results by Agent:
"""
                    for i, task in enumerate(tasklist):
                        combined_text += f"\nAgent {i+1}: {agent_configs[i]['name']}\n"
                        combined_text += f"Role: {agent_configs[i]['role']}\n"
                        combined_text += f"Output:\n{task.output.exported_output}\n"
                        combined_text += "-" * 50 + "\n"
                    
                    # Store in session state instead of showing directly
                    st.session_state.download_content = combined_text 