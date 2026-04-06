# System Design

This document converts the thesis design chapters into a project structure that is implementable in code.

## System Overview

The seed-classification system supports two flows:

- A feature-based tabular classifier for the UCI Seeds-style dataset
- An image-based classifier scaffold for the 14-class transfer-learning thesis variant

## DFD Level 0

```mermaid
flowchart LR
    user["User / Researcher"] --> system["Seed Classification System"]
    system --> result["Seed Class + Metrics + Saved Model"]
```

## DFD Level 1

```mermaid
flowchart LR
    user["User"] --> ingest["Dataset Ingestion"]
    ingest --> prep["Preprocessing"]
    prep --> train["Training"]
    train --> evaluate["Evaluation"]
    evaluate --> artifacts["Artifacts and Reports"]
    artifacts --> user
```

## DFD Level 2

```mermaid
flowchart TD
    raw["Raw seed dataset"] --> validate["Validate schema"]
    validate --> split["Train / test split"]
    split --> scale["Feature scaling"]
    scale --> models["Model training"]
    models --> compare["Model comparison"]
    compare --> metrics["Accuracy, precision, recall, F1, confusion matrix"]
    metrics --> save["Persist model and reports"]
```

## Use Case Diagram

```mermaid
flowchart LR
    actor["User"]
    uc1["Train tabular model"]
    uc2["Predict seed class"]
    uc3["Review metrics"]
    uc4["Prepare image dataset"]
    uc5["Train VGG16 model"]

    actor --> uc1
    actor --> uc2
    actor --> uc3
    actor --> uc4
    actor --> uc5
```

## Activity Flow

```mermaid
flowchart TD
    start["Start"] --> choose["Choose thesis mode"]
    choose --> tabular["Tabular seed classification"]
    choose --> image["Image seed classification"]
    tabular --> load["Load dataset"]
    load --> prep["Preprocess features"]
    prep --> train["Train selected model"]
    train --> eval["Evaluate results"]
    eval --> done["Save artifacts and finish"]
    image --> layout["Validate image folders"]
    layout --> plan["Build transfer-learning configuration"]
    plan --> done
```

## Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant C as CLI Script
    participant D as Dataset Loader
    participant M as Model
    participant E as Evaluator

    U->>C: Start training
    C->>D: Load dataset
    D-->>C: Features and labels
    C->>M: Fit model
    M-->>C: Trained parameters
    C->>E: Compute metrics
    E-->>C: Accuracy, F1, confusion matrix
    C-->>U: Save artifacts and report
```

## Feasibility Summary

- Technical feasibility: high for the tabular pipeline because it runs on NumPy and pandas only
- Economic feasibility: low entry cost because the current implementation avoids heavyweight infrastructure
- Operational feasibility: high because command-line scripts allow straightforward training and prediction
- Future feasibility: the image pipeline can be activated once the 14-class dataset and TensorFlow environment are available
