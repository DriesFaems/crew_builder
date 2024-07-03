
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


client = Groq()

# create title for the streamlit app

st.title('Autonomous Crew Builder')

# create a description for the streamlit app

st.write('This app allows you to create an autonomous crew of agents that can work together to achieve a common goal. You need to define upfront the number of agents that you will use. The agents will work in a sequential order. The agents can be assigned different roles, goals, backstories, tasks and expected outputs. The agents will work together to achieve the common goal. The app will display the output of each agent after the crew has completed its work.')

# ask for the API key in password form
groq_api_key = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = groq_api_key



GROQ_LLM = ChatGroq(
            # api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-70b-8192"
        )

# ask user in streamlit to enter the number of agents that should be part of the crew

number_of_agents = st.number_input('Enter the number of agents that should be part of the crew', min_value=1, max_value=10, value=1)

namelist = []
rolelist = []
goallist = []
backstorylist = []
taskdescriptionlist = []
outputlist = []
toollist = []

for i in range(0,number_of_agents):
    # ask user in streamlit to enter the name of the agent
    question = 'Enter the name of agent ' + str(i+1)
    agent_name = st.text_input(question)
    namelist.append(agent_name)
    role = st.text_input(f"""Enter the role of agent {agent_name}""")
    rolelist.append(role)
    goal = st.text_input(f"""Enter the goal of agent {agent_name}""")
    goallist.append(goal)
    backstory = st.text_input(f"""Describe the backstory of agent {agent_name}""")
    backstorylist.append(backstory)
    taskdescription = st.text_input(f"""Describe the task of agent {agent_name}""")
    taskdescriptionlist.append(taskdescription)
    output = st.text_input(f"""Describe the expected output of agent {agent_name}""")
    outputlist.append(output)
    

# create click button 

if st.button('Create Crew'):
    agentlist = []
    tasklist = []
    for i in range(0, number_of_agents):
        if len(toollist) == 0:
            agent = Agent(
                role=rolelist[i],
                goal=goallist[i],
                backstory=backstorylist[i],
                llm=GROQ_LLM,
                verbose=True,
                allow_delegation=False,
                max_iter=5,
                memory=True
            )
            agentlist.append(agent)
        task = Task(
            description=taskdescriptionlist[i],
            expected_output=outputlist[i],
            agent=agent
        )
        tasklist.append(task)
    crew = Crew(
        agents=agentlist,
        tasks=tasklist,
        verbose=2,
        process=Process.sequential,
        full_output=True,
        share_crew=False,
    )
    # Kick off the crew's work
    results = crew.kickoff()
    # Print the results
    st.write("Crew Work Results:")
    for i in range(0, number_of_agents):
        st.write(f"Agent {i+1} output: {tasklist[i].output.raw_output}")
else:
    st.write('Please click the button to perform an operation')
