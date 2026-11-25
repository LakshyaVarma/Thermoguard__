# ThermoGuard Motor

A neural-network **virtual temperature sensor** for an electric motor. It estimates the internal temperatures of four motor parts from ordinary operating measurements, so the temperatures can still be tracked if a physical sensor fails or isn't installed.

## Why this exists

Temperatures inside a motor (especially the rotor magnet) are hard to measure directly. Sensors there are costly, fragile or impossible to fit. Overheating can demagnetise the magnets or damage the winding insulation, so a trustworthy estimate matters. ThermoGuard predicts these temperatures from signals the motor controller already has, and it keeps working when one of those signals drops out.

## What it predicts

| Output | Component |
|---|---|
| `pm` | Rotor magnet |
| `stator_winding` | Stator winding |
| `stator_tooth` | Stator tooth |
| `stator_yoke` | Stator yoke |

## What it uses as input

Seven measurements:

| Input | Meaning |
|---|---|
| `i_d`, `i_q` | d- and q-axis current |
| `u_d`, `u_q` | d- and q-axis voltage |
| `motor_speed` | Rotor speed |
| `coolant` | Coolant temperature |
| `ambient` | Ambient temperature |

These are expanded into **49 features**: current readings, derived quantities (current squared, speed magnitude, power proxy), moving averages over 10 s, 1 min, 5 min and 15 min (because motors heat and cool slowly), sensor-availability flags, time since each sensor was last seen, and elapsed time in the drive. **Measured target temperatures are never used as inputs.**

## Handling sensor failure

- **Missing readings:** if an input stops reporting, its last value is held for 20 seconds. After that, the training median is used instead.
- **Failure-aware training:** every training drive is used twice, once with healthy inputs and once with the coolant reading removed for 120 s partway through the drive. The model learns to cope with a failed coolant sensor.
- **Availability flags:** the model is told which sensors are currently reporting, and for how long each has been silent.
- **Limitation:** ambient temperature must always be available. The code raises an error if it isn't. Failure robustness was trained and tested for the coolant sensor only, not for the other inputs.

## Model

A small fully connected network: **49 → 96 → 64 → 4** (ReLU activations, about 11,000 parameters). It predicts temperature rise above ambient, scaled by 50. Training uses AdamW with early stopping (stops after 10 epochs without improvement) and keeps the best validation weights. The saved file is small enough for embedded or edge deployment.

## Dataset

[Electric Motor Temperature](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature) (Kirchgässner, Wallscheid and Böcker, Paderborn University), licensed CC BY-SA 4.0. It holds 69 measured drives sampled every 0.5 s.

Drives are split whole (never mid-drive): **48 train / 7 validation / 14 test**, with a fixed random seed. This ensures the model is tested on drives it has never seen.

## Notebooks and sampling rate

All notebooks are identical except for how much the 0.5 s data is downsampled:

| Notebook | Rows kept | Sample interval |
|---|---|---|
| `ThermoGuard_Training_one-third.ipynb` | every 3rd | 1.5 s |
| `01_Train_Model_Colab.ipynb` | every 4th | 2.0 s |
| `ThermoGuard_Training_one-sixth.ipynb` | every 6th | 3.0 s |

Choose the interval that matches how often your deployed system will actually sample. A model trained at one interval should be used at that same interval, because its time-based features depend on it.

## Results

Evaluated on the 14 held-out test drives.

| Model | Validation MAE (°C) | Test MAE (°C) | Test R² |
|---|---|---|---|
| 1.5 s (one-third) | _TBD_ | _TBD_ | _TBD_ |
| 2.0 s (one-fourth) | _TBD_ | _TBD_ | _TBD_ |
| 3.0 s (one-sixth) | _TBD_ | _TBD_ | _TBD_ |

MAE is the average absolute temperature error, with each test drive weighted equally. Errors tend to be larger at the start of a drive, because the initial internal temperature is unknown.

## How to use

1. Open a notebook in Google Colab and run all cells in order (Runtime → Run all).
2. The notebook downloads the dataset, trains the network and evaluates it on the test drives.
3. It saves `thermoguard_model.pt` (weights, input scaling, fallback values and split information) and downloads it.
4. The saved file is then loaded by the follow-up notebook (02) for demonstration and inference.

## Limitations

- Trained on one motor and one dataset. Accuracy on a different motor design or operating range isn't guaranteed.
- Only coolant-sensor dropout was trained and tested.
- Results are estimates, so it should not be the sole safeguard for protecting a motor without further validation.

## Credits and licence

Dataset: Kirchgässner, W., Wallscheid, O., Böcker, J., Paderborn University (CC BY-SA 4.0).
