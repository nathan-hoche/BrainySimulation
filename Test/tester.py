import os
import sys
import importlib

sys.path.append(os.getcwd())
if os.getcwd().endswith("Test"):
    sys.path.append(os.getcwd() + "/../")

CURRENT_FOLDER = os.getcwd()
if not CURRENT_FOLDER.endswith("Test"):
    CURRENT_FOLDER += "/Test/"

def log(func):
    def inner():
        try:
            fd = open(CURRENT_FOLDER + "/test.log", "w")
        except:
            fd = open(CURRENT_FOLDER + "/test.log", "x")
        old_stdout = sys.stdout
        sys.stdout = fd
        plugin_list = func()
        sys.stdout.close
        sys.stdout = old_stdout
        return plugin_list
    return inner

@log
def loadAllTest() -> object:
    MODULE = {}
    MODULE_CRASHED = []
    for dir in os.listdir(CURRENT_FOLDER):
        sys.path.append(CURRENT_FOLDER + "/" + dir + "/")
        if not os.path.isdir(CURRENT_FOLDER + "/" + dir) or dir == "__pycache__":
            continue
        print("Load: ", dir, "found.")
        for filename in os.listdir(CURRENT_FOLDER + "/" + dir + "/"):
            if filename.endswith(".py"):
                try:
                    fd = importlib.import_module(filename.replace(".py", ""))
                    print("\tLoad: ", filename, "found.")
                    classfd = fd.test()
                    print("\tLoad: Class found. -> ", type(classfd))
                    classfd.test(True)
                    MODULE[filename] = classfd
                except Exception as e:
                    print("\tERROR: class/method crashed in", filename, end="\n\t")
                    print(e)
                    MODULE_CRASHED.append(filename)
    return MODULE, MODULE_CRASHED

def main():
    MODULE, MODULE_CRASHED = loadAllTest()
    TOTAL = {"success": 0, "failed": 0, "total": 0, "crashed": 0}
    for module in MODULE:
        print("Test: ", module, "\n")
        result = MODULE[module].test()
        print("\nRESULT:", result)
        for key in result.keys():
            TOTAL[key] += result[key]
        print("\n=======================\n")
    print("FINAL RESULT:", TOTAL)

if __name__ == "__main__":
    main()
