
# Creating a new repo in github
# Trying to connect to github to create a new repo, from chatgpt
# To add a file to a repo from google colab cell add this text to the top of the cell
############## %%writefile name_of_file.py
import requests
import json


# Replace with your GitHub username and personal access token
GITHUB_USERNAME = 'Username'
GITHUB_TOKEN = 'Token'

# Set up the repository details
repo_name = "Colab to github integration"   # Modify these
repo_description = "Code to do stuff in github from colab"
private = False  # Set to True if you want a private repo

# Create the request body
data = {
    "name": repo_name,
    "description": repo_description,
    "private": private
}

# Make the POST request to the GitHub API
response = requests.post(
    f'https://api.github.com/user/repos',
    headers={'Authorization': f'token {GITHUB_TOKEN}'},
    data=json.dumps(data)
)

# Check the response status
if response.status_code == 201:
    print(f"Repository '{repo_name}' created successfully.")
else:
    print(f"Failed to create repository: {response.json()}")
