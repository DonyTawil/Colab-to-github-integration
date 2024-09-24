!git add github_new_file.py # Or !git add github_existing_modified_file.py

!git commit -m "Added in the comments the line that allow you to write a file.py to commit to the repo from google colab"

git_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{git_repo}.git" 

# Push changes to GitHub (replace TOKEN and USERNAME)

!git remote set-url origin git_url

!git push origin Main

# In case of mistake

!git rebase -i commit_number # Commit number I can find looking at errors or !git log, didn't figure this out fully

!git fetch origin                 # But this did work
!git reset --hard origin/Main     # However this deletes all new local files
