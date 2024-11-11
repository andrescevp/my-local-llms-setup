"""
Class that clone a git repository in var folder
"""
import logging
import os
import subprocess


class RepoCloner:
    def __init__(self, repo_url, repo_name, path=f"{os.getenv('PWD')}/var"):
        self.repo_url = repo_url
        self.repo_name = repo_name
        self.repo_path = path + '/' + repo_name
        self.logger = logging.getLogger(__name__)

    def clone(self):
        # if os.path.exists(self.repo_path):
        #     shutil.rmtree(self.repo_path)
        os.makedirs(self.repo_path)
        self.logger.info(f"Cloning {self.repo_url} into {self.repo_path}")
        subprocess.run(['git', 'clone', self.repo_url, self.repo_path])
        self.logger.info(f"Cloned {self.repo_url} into {self.repo_path}")
        return self.repo_path
