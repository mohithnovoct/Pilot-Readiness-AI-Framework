# Creating Custom Plugins

The framework is designed to be extended.  Every pipeline stage has
an abstract base class (ABC) that you can subclass to bring your own
sensors, features, models, fusion strategies, and alert handlers.

---

## Architecture Overview

```
Sensor → Feature Extractor → Model → Fusion → Threshold → Alert
  ↑            ↑               ↑        ↑          ↑          ↑
 ABC          ABC             ABC      ABC        ABC        ABC
```

All ABCs are in `src/core/base.py`.

---

## 1. Custom Sensor Adapter

```python
from src.core.base import BaseSensorAdapter, SensorData
import numpy as np

class MyEEGSensor(BaseSensorAdapter):
    name = "my_eeg"

    def load(self, file_path: str, **kwargs) -> SensorData:
        # Load your data from any format
        eeg_data = np.load(file_path)

        return SensorData(
            signals={"EEG_alpha": eeg_data[:, 0], "EEG_beta": eeg_data[:, 1]},
            sampling_rates={"EEG_alpha": 256.0, "EEG_beta": 256.0},
            labels=None,
            subject_id="pilot_001",
            metadata={"device": "OpenBCI"},
        )

    def stream(self, callback, **kwargs):
        """Optional: implement for live streaming."""
        import serial
        ser = serial.Serial(kwargs.get("port", "/dev/ttyUSB0"), 115200)
        while True:
            chunk = np.frombuffer(ser.read(512), dtype=np.float32)
            callback(chunk)
```

### Register and Use

```python
from src import PilotReadinessFramework

fw = PilotReadinessFramework()
fw.register_sensor(MyEEGSensor())

# Now "my_eeg" is available as a sensor
print(fw.list_components())
```

---

## 2. Custom Feature Extractor

```python
from src.core.base import BaseFeatureExtractor, FeatureSet, SensorData
import pandas as pd
import numpy as np

class EEGBandPowerExtractor(BaseFeatureExtractor):
    name = "eeg_bands"
    feature_names = ["alpha_power", "beta_power", "theta_power", "alpha_beta_ratio"]

    def extract(self, data: SensorData, **kwargs) -> FeatureSet:
        from scipy.signal import welch

        features = []
        for channel in ["EEG_alpha", "EEG_beta"]:
            if channel in data.signals:
                signal = data.signals[channel]
                fs = data.sampling_rates[channel]
                freqs, psd = welch(signal, fs=fs)
                # Compute band powers...
                features.append({"alpha_power": ..., "beta_power": ...})

        return FeatureSet(
            dataframe=pd.DataFrame(features),
            feature_names=self.feature_names,
        )
```

---

## 3. Custom Model

```python
from src.core.base import BaseModel
import numpy as np

class TransformerStressModel(BaseModel):
    name = "transformer_stress"

    def __init__(self):
        self.model = None

    def train(self, X, y, **kwargs):
        import torch
        # Build and train your PyTorch model...
        self.model = ...
        return {"accuracy": 0.95, "f1": 0.93}

    def predict(self, X):
        import torch
        with torch.no_grad():
            return self.model(torch.tensor(X)).numpy()

    def save(self, path):
        import torch
        torch.save(self.model.state_dict(), path)
        return path

    def load(self, path):
        import torch
        self.model.load_state_dict(torch.load(path))
```

---

## 4. Custom Fusion Engine

```python
from src.core.base import BaseFusionEngine
import numpy as np

class AttentionFusion(BaseFusionEngine):
    name = "attention"

    def __init__(self):
        self.attention_weights = {}

    def fuse(self, scores, weights=None, **kwargs):
        # Learn attention weights from recent predictions
        values = np.array(list(scores.values()))
        # Softmax attention
        exp_v = np.exp(values)
        attention = exp_v / exp_v.sum()
        return float(np.dot(attention, values))
```

---

## 5. Custom Alert Handler

```python
from src.core.base import BaseAlertHandler, AlertLevel, PredictionResult

class CockpitAudioAlert(BaseAlertHandler):
    name = "cockpit_audio"

    def handle(self, alert_level, prediction, **kwargs):
        if alert_level == AlertLevel.CRITICAL:
            self._play_sound("critical_alert.wav")
        elif alert_level == AlertLevel.WARNING:
            self._play_sound("warning_tone.wav")

    def should_fire(self, alert_level):
        return alert_level.value >= AlertLevel.WARNING.value

    def _play_sound(self, filename):
        import subprocess
        subprocess.Popen(["aplay", filename])
```

---

## Registering Plugins

### Programmatic Registration

```python
fw = PilotReadinessFramework()
fw.register_sensor(MyEEGSensor())
fw.register_extractor(EEGBandPowerExtractor())
fw.register_model(TransformerStressModel())
fw.register_fusion(AttentionFusion())
fw.register_alert_handler(CockpitAudioAlert())
```

### Via the Registry (advanced)

```python
from src.core.registry import registry

registry.register("sensor", MyEEGSensor())
registry.register_class("model", TransformerStressModel)  # lazy instantiation
```
