import warnings
from bias_estimation import SelectNBiasedWords

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    datapath = "data/askmen.csv" # 7000 Comments
    cl1, cl2 = SelectNBiasedWords(data_path=datapath)
    
    print(cl1, cl2)
    