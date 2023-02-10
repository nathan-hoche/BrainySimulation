from BrainV2.layers.MaxPool1D import MaxPool1D

class test():
    def __init__(self) -> None:
        self.nets = {}
        self.nets["Basic"] = {"net": MaxPool1D(1, pool_size=2), "result": [1, 3, 5, 7, 9, 11, 13, 15]}

        self.INPUT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    def test(self, isTest:bool=False):
        if isTest == True:
            return
        result = {"success": 0, "failed": 0, "total": len(self.nets), "crashed": 0}
        for key in self.nets:
            try:
                res = self.nets[key]["net"].calc(self.INPUT)
                if res.tolist() == self.nets[key]["result"]:
                    print(key + ": success.")
                    result["success"] += 1
                else:
                    print(key + ": failed.")
                    print("Expected: " + str(self.nets[key]["result"]))
                    print("Got:", res.tolist())
                    result["failed"] += 1
            except Exception as e:
                print(key + ": failed.")
                print("Error: " + str(e))
                result["crashed"] += 1
        return result
        

