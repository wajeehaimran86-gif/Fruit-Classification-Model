
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),"models"))
from robust_pipeline import training_features
from svm import OvRSVM

DATA_DIR=os.path.join("data","processed","train")
CACHE=os.path.join("results","saved_models")
MODEL_OUT=os.path.join(CACHE,"linear_svm_robust.npz")

def main():
    Xs=[]; ys=[]
    classes=sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))])
    print("Robust training classes:", classes)
    for cls in classes:
        cdir=os.path.join(DATA_DIR,cls)
        files=[f for f in os.listdir(cdir) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        print(f"{cls}: {len(files)} base images -> {len(files)*6} robust views")
        for j,f in enumerate(files):
            try:
                feats=training_features(os.path.join(cdir,f), seed_base=42+j*17)
                Xs.append(feats)
                ys.extend([cls]*len(feats))
            except Exception as e:
                print("[skip]",f,e)
    X=np.vstack(Xs).astype(np.float32); y=np.array(ys)
    print("Robust feature matrix:",X.shape)
    os.makedirs(CACHE,exist_ok=True)
    np.savez(os.path.join(CACHE,"robust_train_features.npz"),X=X,y=y)

    # Balance classes without changing the original dataset: undersample only
    # the largest class to the median class size.
    counts={c:int(np.sum(y==c)) for c in classes}
    target=int(np.median(list(counts.values())))
    rng=np.random.default_rng(42)
    idx=[]
    for c in classes:
        ci=np.where(y==c)[0]
        if len(ci)>target: ci=rng.choice(ci,target,replace=False)
        idx.extend(ci.tolist())
    idx=np.array(idx)
    rng.shuffle(idx)
    Xb,yb=X[idx],y[idx]
    print("Balanced robust matrix:",Xb.shape, {c:int(np.sum(yb==c)) for c in classes})

    model=OvRSVM(learning_rate=0.0005,C=1.0,n_epochs=400)
    model.fit(Xb,yb)
    train_pred=model.predict(Xb)
    print("Robust augmented training accuracy:",np.mean(train_pred==yb))
    model.save(MODEL_OUT)
    print("Saved:",MODEL_OUT)

if __name__=="__main__":
    main()
