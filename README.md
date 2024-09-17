![SemanticLangBanner](https://res.cloudinary.com/dn3o5dlna/image/upload/v1726419733/banners/SemanticLangPoster.png)
_Designed by me :3 (This image's not CC btw)_
# SemanticLang
_A lightweight Q&A model designed to answer questions about long documents, with traceable deduction and evidence_

### Special tokens
- \<unused0>: \<T> (Start of Triple (Subject))
- \<unused1>: \<R> (Predicate)
- \<unused2>: \<S> (Object)

### pre fine-tuned models
We have included the most recent (submission) model weights in the repository.  
You may also train your own using the x_2b_train.ipynb documents.  
Training data is included

### General usage
Run all in complete_integration.ipynb and do as prompted. Everything else should work automatically. If you just want the answer (you probably don't) go to the bottom of the document. The final answer will be printed there.  
Intermediate steps are printed and visualized throughout the document.
