import pickle
import os
from glob import glob


def main():
    pickle_files = sorted(glob(os.path.join("./output/", "*.pkl")))

    for pf in pickle_files:
        basename = os.path.basename(pf)

        with open(pf, "rb") as f:     
             po = pickle.load(f)
             fixed_po = [[int(v) for v in l] for l in po]
            
        with open(f"./output/{basename}_fixed.pkl", "wb") as f:
            pickle.dump(fixed_po, f)


if __name__ == "__main__":
    main()