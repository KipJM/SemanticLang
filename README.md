![SemanticLangBanner](https://res.cloudinary.com/dn3o5dlna/image/upload/v1726419733/banners/SemanticLangPoster.png)
_Designed by me :3 (This image's not CC btw)_
# SemanticLang
_A lightweight Q&A model designed to answer questions about long documents, with traceable deduction and evidence_

### Special tokens

#### GraphGen
- ```<unused0>: <T> (Start of Triple (Subject))```
- ```<unused1>: <R> (Predicate)```
- ```<unused2>: <S> (Object)```
- ```<eos>:         End of String```

#### QueryGen
- ```<unused10>: < (Keyword Start)```
- ```<unused11>: >  (Keyword End)```
- ```<unused12>: {  (Pattern Start)```
- ```<unused13>: }  (Pattern End)```
- ```<unused14>: ?  (Variable)```
- ```<unused15>: .  (Seperator)```

### pre fine-tuned models
We have included the most recent (submission) model weights in the repository.  
You may also train your own using the x_2b_train.ipynb documents.  
Training data is included.

### 9b models (experimental)
Additionally, we fine-tuned 1 epoch models based on Gemma-2-9b. These models will not be part of the git repo (too big), but they can be found here(TBA).

graph-9b-full is fine-tuned on an Nvidia A100 for (4? haha idk it's still cooking) days. They have much higher accuracy than the previous demo model (graph-2b-1000step)

### General usage
Run all in ``complete_integration.ipynb`` and respond as prompted. Everything else should work automatically. If you just want the answer (you probably don't) go to the bottom of the notebook. The final answer will be printed there.  

Intermediate steps are printed and visualized throughout the document.  

``Graph.html`` is the generated graph if Jupyter breaks and refuses to display graphs again


