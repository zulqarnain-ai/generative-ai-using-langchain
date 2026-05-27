# from langchain_community.tools import DuckDuckGoSearchRun

# search_tool = DuckDuckGoSearchRun()

# results = search_tool.invoke('today date ')

# print(results)

#----------------------------------------------------------

# from langchain_community.tools import ShellTool

# tool= ShellTool()

# result = tool.invoke('whoami')

# ---------------------------------------------------------------

# from langchain_community.tools import ShellTool

# tool = ShellTool()

# # Use a dictionary with a "commands" key containing a list of strings
# result = tool.invoke({"commands": ["ls"]})

# print(result)

# ----------------------------------------------------------

import subprocess
from langchain_core.tools import tool

@tool
def native_shell_tool(command: str) -> str:
    """Executes a shell command on the local machine and returns the output. 
    Use with extreme caution."""
    try:
        # Run the command and capture the output
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr}"

# How to invoke your new native tool
result = native_shell_tool.invoke({"command": "mkdir dir"})
print(result)