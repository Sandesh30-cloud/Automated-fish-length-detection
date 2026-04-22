# Project UML

This document provides UML-style diagrams for the current merged project workspace.

Scope covered:

- `matsyavan/`
- `Major Project/`
- `Fish Growth Prediction/`
- `additional/Sensors_model/`
- `ml/`

The diagrams below use Mermaid so they can be rendered directly in compatible Markdown viewers.

## 1. High-Level Component Diagram

```mermaid
flowchart LR
    User["User / Fish Farmer"]
    Frontend["Matsyavan Frontend\nReact + TypeScript + Vite"]
    Backend["Matsyavan Backend\nNode.js + Express + WebSocket"]
    LocalDB["Local JSON Database\nmatsyavan/data/local-db.json"]
    Firebase["Firebase\nAuth + Realtime Database"]
    Vision["Major Project\nFish Detection + Homography Measurement"]
    VisionCSV["measurements.csv\nFish Length Output"]
    GRU["Fish Growth Prediction\nGRU Model"]
    ML["ml/\nOther ML Scripts"]
    SensorsModel["additional/Sensors_model\nWater Quality Classifier"]
    Device["ESP32 / Sensor Source"]

    User --> Frontend
    Frontend --> Backend
    Frontend --> Firebase
    Device --> Backend
    Device --> Firebase
    Backend --> LocalDB
    Vision --> VisionCSV
    Backend --> VisionCSV
    Backend --> GRU
    Backend --> ML
    Backend --> SensorsModel
    Firebase --> Frontend
```

## 2. Deployment Diagram

```mermaid
flowchart TB
    subgraph ClientSide["Client Side"]
        Browser["Browser\nDashboard UI"]
    end

    subgraph LocalMachine["Local Machine / Development Node"]
        FrontendApp["Vite Frontend\nlocalhost:5173"]
        ApiServer["Express API + WS\n127.0.0.1:8787"]
        JsonStore["Local JSON Store"]
        VisionPipeline["Major Project Pipeline"]
        GrowthModel["Fish Growth Prediction\nGRU .keras model"]
        OtherML["Other Python ML Scripts"]
    end

    subgraph Cloud["Cloud Services"]
        FirebaseAuth["Firebase Auth"]
        FirebaseRTDB["Firebase RTDB\nWaterQuality/current"]
    end

    subgraph Edge["Edge / Device Layer"]
        Esp32["ESP32 / IoT Sensors"]
    end

    Browser --> FrontendApp
    FrontendApp --> ApiServer
    FrontendApp --> FirebaseAuth
    FrontendApp --> FirebaseRTDB
    Esp32 --> ApiServer
    Esp32 --> FirebaseRTDB
    ApiServer --> JsonStore
    ApiServer --> VisionPipeline
    ApiServer --> GrowthModel
    ApiServer --> OtherML
```

## 3. Matsyavan Internal Class Diagram

```mermaid
classDiagram
    class App {
      +espData
      +renderContent()
    }

    class Dashboard {
      +renderMetrics()
      +predictGrowth()
    }

    class Reports {
      +loadHistory()
      +predictGrowth()
      +generateReport()
    }

    class LocalApiService {
      +getCurrentData()
      +getHistory()
      +predictGrowth()
      +predictWaterQuality()
      +connectWebSocket()
      +subscribeToData()
    }

    class Routes {
      +GET /api/data/current
      +GET /api/data/history
      +POST /api/ml/predict-growth
      +POST /api/ml/predict-water-quality
      +POST /api/measurements
    }

    class LocalDatabase {
      +saveMeasurement(payload)
      +updateCurrent(partial)
      +getCurrent()
      +getHistory()
      +createUser()
    }

    class MLPredictor {
      +predictTilapiaGrowth(input)
      +predictWaterQualityStatus(input)
      +runPythonPrediction(scriptPath,args)
    }

    class FishMeasurementSync {
      +syncLatestFishMeasurement()
    }

    App --> Dashboard
    App --> Reports
    Dashboard --> LocalApiService
    Reports --> LocalApiService
    LocalApiService --> Routes
    Routes --> LocalDatabase
    Routes --> MLPredictor
    Routes --> FishMeasurementSync
```

## 4. Fish-Length Measurement Workflow Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User / Operator
    participant Img as Image Folder / Camera
    participant Auto as automation_controller.py
    participant Presence as fish-present-or-not/inference.py
    participant Homo as homography.py / fish_measurement
    participant CSV as results/measurements.csv
    participant API as Matsyavan API
    participant UI as Dashboard

    U->>Img: Capture or place fish image
    Img->>Auto: New image available
    Auto->>Presence: Detect fish presence
    Presence-->>Auto: fish_present / confidence
    alt Fish detected
        Auto->>Homo: Run homography-based measurement
        Homo-->>Auto: fish_count + lengths_mm + overlay_path
        Auto->>CSV: Append measurement row
        API->>CSV: Read latest row
        API-->>UI: fishLengthCm / fishLengthMm
    else No fish detected
        Auto->>CSV: Append negative/no-measurement result
    end
```

## 5. Fish Growth Prediction Sequence Diagram

```mermaid
sequenceDiagram
    participant UI as Matsyavan Dashboard
    participant API as POST /api/ml/predict-growth
    participant Predictor as mlPredictor.js
    participant GRU as Fish Growth Prediction/predict.py
    participant Model as GRU.keras
    participant Data as final_dataset_with_growth.csv

    UI->>API: Request growth prediction
    API->>Predictor: predictTilapiaGrowth({})
    Predictor->>GRU: Spawn python3 predict.py --json
    GRU->>Data: Load processed dataset
    GRU->>Model: Load GRU model
    GRU->>GRU: Build sequence (dataset tail or explicit sequence)
    GRU->>GRU: Predict next weight difference
    GRU-->>Predictor: JSON predicted_weight
    Predictor-->>API: Parsed prediction
    API-->>UI: predictedWeight + metadata
```

## 6. Sensor Ingestion Sequence Diagram

```mermaid
sequenceDiagram
    participant ESP as ESP32 / Sensor Source
    participant API as Matsyavan API
    participant DB as LocalDatabase
    participant WS as WebSocket Server
    participant UI as Frontend
    participant RTDB as Firebase RTDB

    ESP->>API: POST /api/measurements
    API->>DB: saveMeasurement(payload)
    DB-->>API: current + history updated
    API->>WS: broadcastData(current)
    WS-->>UI: real-time sensor update

    opt Optional Firebase path
        ESP->>RTDB: Write WaterQuality/current
        RTDB-->>UI: Firebase realtime listener update
    end
```

## 7. Fish Growth Prediction Module Class Diagram

```mermaid
classDiagram
    class PredictPy {
      +parse_args()
      +build_sequence()
      +estimate_weight_from_length()
      +main()
    }

    class DataLoader {
      +load_data(path)
    }

    class Preprocessing {
      +fit_feature_scaler(values)
      +fit_target_scaler(values)
      +scale_features(sequence,min,span)
      +inverse_scale_target(value,min,span)
    }

    class GRUModel {
      +GRU.keras
    }

    PredictPy --> DataLoader
    PredictPy --> Preprocessing
    PredictPy --> GRUModel
```

## 8. Use Case Diagram

```mermaid
flowchart LR
    Farmer["Fish Farmer / User"]
    Admin["System Admin / Developer"]

    UC1["View live water-quality dashboard"]
    UC2["View fish length"]
    UC3["Get fish growth prediction"]
    UC4["Generate reports"]
    UC5["Replay sensor CSV to Firebase"]
    UC6["Run image-based fish measurement"]
    UC7["Authenticate with Firebase"]

    Farmer --> UC1
    Farmer --> UC2
    Farmer --> UC3
    Farmer --> UC4
    Farmer --> UC7

    Admin --> UC5
    Admin --> UC6
    Admin --> UC7
```

## 9. Package / Module Relationship Diagram

```mermaid
flowchart TD
    Root["merged/"]

    Root --> Matsyavan["matsyavan/"]
    Root --> Major["Major Project/"]
    Root --> Growth["Fish Growth Prediction/"]
    Root --> ML["ml/"]
    Root --> Sensors["additional/Sensors_model/"]

    Matsyavan --> MatsyavanSrc["src/"]
    Matsyavan --> MatsyavanScripts["scripts/"]
    Matsyavan --> MatsyavanData["data/"]

    Major --> MajorMeasure["fish_measurement/"]
    Major --> MajorPresence["fish-present-or-not/"]
    Major --> MajorResults["results/"]

    Growth --> GrowthSrc["src/"]
    Growth --> GrowthData["data/processed/"]
    Growth --> GrowthOutputs["outputs/models/"]

    ML --> MLModels["models/"]
    Sensors --> SensorPredict["predict.py"]
```

## Notes

- The current project is hybrid:
  - local API + WebSocket
  - Firebase Auth
  - optional Firebase RTDB
- Fish-length estimation is sourced from `Major Project/results/measurements.csv`
- Fish growth prediction is currently sourced from `Fish Growth Prediction/predict.py`
- Additional ML scripts remain in `ml/` and `additional/Sensors_model/`, even if they are not the primary live dashboard path

## Suggested Usage

For:

- report: use diagrams 1, 4, 5, and 6
- viva/presentation: use diagrams 1, 2, and 8
- developer docs: use diagrams 3, 7, and 9
