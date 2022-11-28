import os
import sys
import importlib

######## Loading Part ########

def load_sample(filename: str) -> object:
    try:
        sys.path.append(os.getcwd() + "/sample/")
        fd = importlib.import_module(filename.replace(".py", ""))
        print("Load: ", filename, "found.")
    except Exception as e:
        print("ERROR: File", filename, "not found.")
        print(e)
        exit(0)
    try:
        classfd = fd.sample()
        print("Load: Class found. -> ", type(classfd))
        ####### Check if sample fonction is set
        classfd.train(True)
        classfd.test(True)
        #######################################
    except Exception as e:
        print("ERROR: class/method crashed")
        print(e)
        exit(0)
    return classfd

######## Main Part ########

def main():
    if len(sys.argv) != 2:
        print("ERROR: Invalid number of arguments.\n Usage: python3 main.py [sample]")
        return
    sample = load_sample(sys.argv[1])
    sample.train()
    sample.test()

main()