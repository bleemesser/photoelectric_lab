# Photoelectricity Lab
My graphing and calculations for the lab.
## Usage:
### Data:
- Must be in CSV format with name `data.csv`.
- Must have at least 2 columns: `Wavelength` and `Voltage`. Wavelength must be in nm and voltage in volts including the negative sign.
- If you choose to include more voltages, they will be averaged. Name them `Voltage 1`, `Voltage 2`, etc.
- See `example.csv` for format, you *must* use that style.
### Prep:
- Place your data.csv in the same folder as `plots.py`.
- With python installed, run in the terminal, in the same directory as you cloned this repository to, `python3 -m pip install -r requirements.txt` to get the needed packages.
- Run `python3 plots.py`
## Sample:
![example](example.png)