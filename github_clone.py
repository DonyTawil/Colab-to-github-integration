# Code to connect and clone the repo
GITHUB_USERNAME = 'TheUsername'
GITHUB_EMAIL = "TheEmail@gmail.com"
GITHUB_REPO = "Colab to github integration"
GITHUB_TOKEN = "the token"

git_user = f"config --global user.name \"{GITHUB_USERNAME}\""
git_email = f"config --global user.email \"{GITHUB_EMAIL}\""
git_repo = GITHUB_REPO.replace(" ", "-")

git_execute = f"clone https://github.com/{GITHUB_USERNAME}/{git_repo}.git"
git_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}/{GITHUB_USERNAME}/{git_repo}.git"

# !git {git_user}
# !git {git_email}
# !git {git_execute} #execute this code to clone the repo

# %cd {git_repo}

# Set up Git configuration
#!git config --global user.name "GITHUB_USERNAME"
#!git config --global user.email "YOUR_EMAIL"

!git add github_clone.py

!git commit -m "Code that allows you to commit new file to github repo"

# Push changes to GitHub (replace GITHUB_TOKEN and GITHUB_USERNAME)
!git remote set-url origin {git_url}
# could do this instead if the variables were correct

!git branch -m Main
!git push -u origin main
