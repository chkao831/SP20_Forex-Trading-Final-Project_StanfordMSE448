# This file was originally developed by AI for Healthcare Bootcamp Winter 2019 
#
# Automatically apply pycodestyle and autopep8.
# E402 error is ignored for our coding style.
# WARNING! This only checks uncommited files.
# You still have to check new files manually.
#
# Usage:
#   # Checks all .py files within the directory
#   bash bash_scripts/py_style_check.py all 
#   # Checks all modified files (since last commit) within the directory
#   bash bash_scripts/py_style_check.py


echo "Operating on path $PWD"

if [ "$1" == "all" ]; then
    files="*/*.py"
else
    echo "Modified files include"
    files=$(git ls-files -m)
fi

echo "Will check modified files"
echo $files

for f in $files
do
    echo "==== Style checking $f ===="
    pycodestyle --ignore E402,E121,E123,E126,E226,E24,E704,W503,W504 $f
    echo "------------------"

    read -p "Apply autopep8? [y/n]" -n 1 -r
    echo 
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        continue
    fi
    autopep8 --in-place --aggressive --aggressive --ignore "E402" $f
    echo "WARNING! autopep8 might not fix all errors. Remaining errors:"
    pycodestyle --ignore E402,E121,E123,E126,E226,E24,E704,W503,W504 $f
    echo "=============================="
done