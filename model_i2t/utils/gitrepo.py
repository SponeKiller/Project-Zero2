import os
import subprocess

class GitRepo:
    def __init__(self, repo_url, dest_folder):
        """
        Initialize GitRepo object
        
        Args:
            repo_url (str): URL of the repository
            dest_folder (str): Destination folder to clone the repository
        """
        
        self.repo_url = repo_url
        self.dest_folder = dest_folder

    def clone(self):
        """
        Clone repository
        """
        currernt_dir = os.getcwd()
        
        if not os.path.isdir(self.dest_folder):
            os.mkdir(self.dest_folder)
            
        os.chdir(self.dest_folder)
        
        result = subprocess.run(
            ["git", "clone", self.repo_url, self.dest_folder], 
            check=True,
            capture_output=True,
            text=True,
        )
        
        os.chdir(currernt_dir)
    
        print("Fetch result:")
        print("Return code:", result.returncode)
        print("Standard output:", result.stdout)
        print("Standard error:", result.stderr)
            
    def fetch(self):
        """
        Fetching changes from repository
        """
        current_dir = os.getcwd()
        
        os.chdir(self.dest_folder)
        
        result = subprocess.run(
            ['git', 'fetch'], 
            check=True, 
            capture_output=True, 
            text=True,
        )
        
        os.chdir(current_dir)
        
        print("Fetch result:")
        print("Return code:", result.returncode)
        print("Standard output:", result.stdout)
        print("Standard error:", result.stderr)


    def pull(self):
        """
        Pull changes from repository
        """
        current_dir = os.getcwd()
        
        os.chdir(self.dest_folder)
        
        subprocess.run(
            ['git', 'fetch'], 
            check=True,
        )
        
        result = subprocess.run(
            ['git', 'pull'], 
            check=True, 
            capture_output=True, 
            text=True,
        )
        
        os.chdir(current_dir)
        
        print("Return code:", result.returncode)
        print("Standard output:", result.stdout)
        print("Standard error:", result.stderr)


    def switch_branch(self, branch_name):
        """
        Switch branch
        
        Args:
            branch_name (str): Name of the branch
        """
        current_dir = os.getcwd()   
        
        os.chdir(self.dest_folder)
        
        result = subprocess.run(
            ["git", "branch", "--list", branch_name], 
            check=True,
            capture_output=True, 
            text=True,
        )
        
        if branch_name in result.stdout:
            print(f"Switching to existing branch: {branch_name}")
            result = subprocess.run(
                ["git", "checkout", branch_name], 
                check=True,
                capture_output=True,
                text=True,
            )
            
        else:
            print(f"Creating new branch: {branch_name}")
            result = subprocess.run(
                ["git", "checkout", "-b", branch_name], 
                check=True,
                capture_output=True,
                text=True,
            )

        os.chdir(current_dir)
        
        print("Switch branch result:")
        print("Return code:", result.returncode)
        print("Standard output:", result.stdout)
        print("Standard error:", result.stderr)


    def push(self):
        """
        Push changes to repository
        """
        current_dir = os.getcwd()
        
        os.chdir(self.dest_folder)
        
        result = subprocess.run(
            ['git', 'push'], 
            check=True, 
            capture_output=True, 
            text=True,
        )
        
        os.chdir(current_dir)
        
        print("Push result:")
        print("Return code:", result.returncode)
        print("Standard output:", result.stdout)
        print("Standard error:", result.stderr)