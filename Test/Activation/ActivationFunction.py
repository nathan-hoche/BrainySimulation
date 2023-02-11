from BrainV2.utils import Activation
import numpy as np

class test():
    def __init__(self) -> None:
        self.func = {}
        self.func["basic"] = {"func": Activation.basic, "result": [0, 1, 1, 1]}
        self.func["reLu"] = {"func": Activation.reLu, "result": [0, 1, 1, 1]}
        self.func["leakyReLu"] = {"func": Activation.leakyReLu, "result": [0, 1, 1, 1]}
        self.func["sigmoid"] = {"func": Activation.sigmoid, "result": [0.5, 0.7310585786300049, 0.7310585786300049, 0.7310585786300049]}
        self.func["tanh"] = {"func": Activation.tanh, "result": [0.0, 0.7615941559557649, 0.7615941559557649, 0.7615941559557649]}
        self.func["softmax"] = {"func": Activation.softmax, "result": [0.10923177257303593, 0.2969227424756547, 0.2969227424756547, 0.2969227424756547]}
        self.func["softplus"] = {"func": Activation.softplus, "result": [0.6931471805599453, 1.3132616875182228, 1.3132616875182228, 1.3132616875182228]}
        self.func["softsign"] = {"func": Activation.softsign, "result": [0.0, 0.5, 0.5, 0.5]}
        self.func["selu"] = {"func": Activation.selu, "result": [0.0, 1.0507, 1.0507, 1.0507]}
        self.func["elu"] = {"func": Activation.elu, "result": [0.0, 1.0, 1.0, 1.0]}
        self.func["exponential"] = {"func": Activation.exponential, "result": [1.0, 2.718281828459045, 2.718281828459045, 2.718281828459045]}

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