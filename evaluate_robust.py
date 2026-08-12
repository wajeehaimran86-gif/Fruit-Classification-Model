
import os,sys,argparse,numpy as np
sys.path.insert(0,os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),"models"))
from robust_pipeline import inference_features
from svm import OvRSVM
from confidence import softmax_confidence

MODEL=os.path.join("results","saved_models","linear_svm_robust.npz")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("image_path")
    args=ap.parse_args()
    if not os.path.exists(args.image_path):
        print("ERROR: Image not found:",args.image_path); return
    model=OvRSVM(); model.load(MODEL)
    views=inference_features(args.image_path)
    scores=model.decision_scores(views)
    # Average normalized scores across the same robustness views.
    probs=softmax_confidence(scores)
    mean_probs=probs.mean(axis=0)
    order=np.argsort(mean_probs)[::-1]
    print("\nImage:",args.image_path)
    print("Robust multi-view probabilities:")
    for i in order:
        print(f"  {model.classes_[i]}: {mean_probs[i]*100:.2f}%")
    best=int(order[0])
    print(f"\nPredicted fruit: {model.classes_[best]}")
    print(f"Confidence: {mean_probs[best]*100:.2f}%")

if __name__=="__main__":
    main()
