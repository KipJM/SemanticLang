# Logic3 - A hybrid AI model for natural language Q&A
Logic3 combines the natural language understanding capabilities of transformers with the performance and reliability of RDF(Resource Description Framework).

## Architecture
```mermaid
flowchart
A[NL Question] --> B(["queryGen (Transformer 1)"]) --> FOL["Semantic relations in RDF"]
B --> FOQ["RDF query"]
subgraph RDF ENGINE
    FOL --> LINK[]
    FOQ --> LINK
    
end
FOL --> FOO[RDF response] --> D(["responseGen (Transformer 2)"])
A --> D --> E["Contextualized RDF response"]
```

## Data
Logic3 is trained on FOLIO, 