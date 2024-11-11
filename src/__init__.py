# check if var folder exists
# if not, create it
import os

from dotenv import load_dotenv

PWD = os.path.dirname(os.path.abspath(__file__))

# set PWD as environment variable
os.environ['PWD'] = PWD

load_dotenv()

if not os.path.exists(f"{PWD}/var"):
    os.makedirs(f"{PWD}/var")
