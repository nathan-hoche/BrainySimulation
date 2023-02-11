from BrainV2.utils import Derivative
import numpy as np

class test():
    def __init__(self) -> None:
        self.func = {}
        self.func["basic"] = {"func": Derivative.basic, "result": [0, 1, 1, 1]}
        self.func["reLu"] = {"func": Derivative.reLu, "result": [0, 1, 1, 1]}
        self.func["leakyReLu"] = {"func": Derivative.leakyReLu, "result": [0, 1, 1, 1]}
        self.func["sigmoid"] = {"func": Derivative.sigmoid, "result": [0, 0, 0, 0]}
        self.func["tanh"] = {"func": Derivative.tanh, "result": [1, 0, 0, 0]}
        self.func["softmax"] = {"func": Derivative.softmax, "result": [0, 0, 0, 0]}
        self.func["softplus"] = {"func": Derivative.softplus, "result": [0.5, 0.7310585786300049, 0.7310585786300049, 0.7310585786300049]}
        self.func["softsign"] = {"func": Derivative.softsign, "result": [1.0, 0.25, 0.25, 0.25]}
        self.func["selu"] = {"func": Derivative.selu, "result": [1.0507, 1.0507, 1.0507, 1.0507]}
        self.func["elu"] = {"func": Derivative.elu, "result": [1.0, 1.0, 1.0, 1.0]}
        self.func["exponential"] = {"func": Derivative.exponential, "result": [2.718281828459045, 2.718281828459045, 2.718281828459045, 2.718281828459045]}

        self.INPUT = np.array([0, 1, 2, 3])

    def test(self, isTest:bool=False):
        if isTest == True:
            return
        result = {"success": 0, "failed": 0, "total": len(self.func), "crashed": 0}
        for key in self.func:
            try:
                res = self.func[key]["func"](self.INPUT)
                if res.tolist() == self.func[key]["result"]:
                    print(key + ": success.")
                    result["success"] += 1
                else:
                    print(key + ": failed.")
                    print("Expected: " + str(self.func[key]["result"]))
                    print("Got:", res.tolist())
                    result["failed"] += 1
            except Exception as e:
                print(key + ": failed.")
                print("Error: " + str(e))
                result["crashed"] += 1
        return result