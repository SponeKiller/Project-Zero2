import subprocess
import builtins

class Util:

    @staticmethod
    def download_package(name: str, from_file: bool = False):
        """
        Downloading package
        
        Args:
            name (str): package name
            from_file (bool): if True, then name is a file path to
                requirements.txt. Default is False
        """
        
        if from_file:
            cmd = ['pip', 'install', '-r', name]
        else:
            cmd = ['pip', 'install', name]
        
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
        )
        print("Download package result:")
        print("Return code:", result.returncode)
        print("Standard output:", result.stdout)
        print("Standard error:", result.stderr)


    @staticmethod
    def is_package_installed(name: str):
        """
        Check if the package is installed
        
        Args:
            name (str): package name
        """
        
        try:
            __import__(name)
            return print(f"{name} is installed")
        except ImportError:
            return print(f"{name} is not installed")