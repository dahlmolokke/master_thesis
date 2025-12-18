"""
This script retrieves VHF radar data from the Madrigal database using the globalIsrint.py script.
Make sure to update the user information and script path before running.
Developed Aug 2025. Madrigal update Sep 2025 requires adjustments.
"""

import subprocess
import os

# Generate paths
script_dir = os.getcwd()
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, "data/raw")

# ----- Constants for user info and madrigal script path -----
MADRIGAL_SCRIPT_PATH = r"/path/to/globalIsrint.py"   # Update this path accordingly
FULL_NAME = "Firstname Lastname"                      # Replace with your full name
USER_EMAIL = "firstname.lastname@domain.com"          # Replace with your email
USER_AFFILIATION = "Your Affiliation"                 # Replace with your affiliation


# Build the command as a list (Windows safe)
cmd = [
    "python", MADRIGAL_SCRIPT_PATH,
    "--verbose",
    "--url=https://madrigal.eiscat.se/madrigal",
    "--parms=YEAR,MONTH,DAY,HOUR,MIN,SEC,GDALT,ELM,AZM,NE,DNE,TI,DTI,TE,DTE,VO,DVO",
    f"--output={data_dir}\\TRO_UHF_2004-2024.txt",
    f"--user_fullname={FULL_NAME}",
    f"--user_email={USER_EMAIL}",
    f"--user_affiliation={USER_AFFILIATION}",
    "--startDate=01/01/2004",
    "--endDate=12/31/2024", 
    "--inst=72",
    "--filter=ELM,70,",
    "--kindat=6400",
    "--expName=beata",
]
subprocess.run(cmd, check=True)
print("Download finished!")