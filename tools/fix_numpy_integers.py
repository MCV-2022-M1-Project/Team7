import pickle
import os
from glob import glob


FOLDER = "./output/qst2_method2"


def main():
    pickle_files = sorted(glob(os.path.join(FOLDER, "result.pkl")))

    for pf in pickle_files:
        basename = os.path.basename(pf)

        with open(pf, "rb") as f:     
             po = pickle.load(f)
             fixed_po = [[[int(v) for v in l[:10]] for l in s] for s in po]
            
        with open(f"{FOLDER}/fixed_{basename}", "wb") as f:
            pickle.dump(fixed_po, f)


if __name__ == "__main__":
    main()