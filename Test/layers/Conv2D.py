from BrainV2.layers.Conv2D import Conv2D, Filter

class test():
    def __init__(self) -> None:
        self.nets = {}
        self.nets["HorizSobel"] = Conv2D(1, Filter.HorizSobel(), (1, 1), padding=True)
        self.nets["VertSobel"] = Conv2D(1, Filter.VertSobel(), (1, 1), padding=True)
        self.nets["Laplacian"] = Conv2D(1, Filter.Laplacian(), (1, 1), padding=True)
        self.nets["Gaussian"] = Conv2D(1, Filter.Gaussian(), (1, 1), padding=True)
        self.nets["Identity"] = Conv2D(1, Filter.Identity(), (1, 1), padding=True)
        self.nets["Sharpen"] = Conv2D(1, Filter.Sharpen(), (1, 1), padding=True)
        self.nets["Emboss"] = Conv2D(1, Filter.Emboss(), (1, 1), padding=True)
        self.nets["BoxBlur"] = Conv2D(1, Filter.BoxBlur(), (1, 1), padding=True)
        self.nets["EdgeDetect"] = Conv2D(1, Filter.EdgeDetect(), (1, 1), padding=True)
        self.nets["EdgeEnhance"] = Conv2D(1, Filter.EdgeEnhance(), (1, 1), padding=True)
        self.nets["MeanRemoval"] = Conv2D(1, Filter.MeanRemoval(), (1, 1), padding=True)
        self.nets["MotionBlur"] = Conv2D(1, Filter.MotionBlur(), (1, 1), padding=True)

        self.INPUT = [[0, 0, 0, 1, 0, 0], [1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 0, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1]]

    def test(self, isTest:bool=False):
        if isTest == True:
            return
        result = {"success": 0, "failed": 0, "total": len(self.nets), "crashed": 0}
        for key in self.nets:
            try:
                _ = self.nets[key].calc(self.INPUT)
                print(key + ": success.")
                result["success"] += 1
            except:
                print(key + ": failed.")
                result["crashed"] += 1
        return result
        

