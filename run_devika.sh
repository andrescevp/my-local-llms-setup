BASE_DIR=$(pwd)/devika-official-repo
cd ./devika-official-repo
# set python version
pyenv local 3.12

# check venv exists
if [ ! -d "venv" ]; then
  python -m venv venv
fi

chmod +x setup.sh
./setup.sh

cd $BASE_DIR

# start devika
python devika.py | cd ui && bun run start