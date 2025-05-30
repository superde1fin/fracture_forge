# Fracture Forge

**Fracture Forge** is a tool for simulating and analyzing fracture propagation in atomistic materials. Built on a custom version of LAMMPS, it identifies probable crack pathways by inserting virtual cutting surfaces, running local energy evaluations, and assembling a likely path based on energetics.

The result is a fracture path that reflects the underlying material structure and interaction potential — useful for estimating energy release rates, crack surface areas, and exploring the stochastic variability of fracture in complex systems.

---

## 🔧 Installation

Fracture Forge includes an installation script to set up the application and dependencies. It also requires a specially compiled version of LAMMPS.

---

### 1. Clone this Repository

```bash
git clone https://github.com/superde1fin/fracture_forge.git
cd fracture_forge
```

---

### 2. Install Fracture Forge

Run the provided configuration script:

```bash
./configure
```

This will:

* Install Python dependencies using `pip`
* Copy required modules into a local folder
* Create an executable named `fforge` under `~/.local/bin` (by default)

#### Optional Flags

* `--prefix=PREFIX` – Install to a custom location (default: `~/.local`)
* `--python=PYTHON_PATH` – Use a specific Python interpreter
* `--help`, `-h` – Display help message

Example:

```bash
./configure --prefix=/opt/fracture_forge --python=/usr/bin/python3.10
```

After installation, add the binary path to your `PATH` if needed by adding the following line to your \~/.bashrc:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

---

### 3. Build Required LAMMPS Version

Fracture Forge depends on a custom version of LAMMPS with the `GHOSTWALL` package and Python bindings enabled.

#### Quick LAMMPS Build Instructions:

```bash
git clone https://github.com/superde1fin/lammps.git
cd lammps
mkdir build && cd build

cmake ../cmake \
  -D BUILD_MPI=on \
  -D BUILD_OMP=on \
  -D BUILD_LIB=on \
  -D BUILD_SHARED_LIBS=on \
  -D PKG_GHOSTWALL=on \
  -D CMAKE_INSTALL_PREFIX=/your/installation/path

cmake --build . -j $(nproc)

make install-python
make install
```

📚 Official guide: [https://docs.lammps.org/Build\_cmake.html](https://docs.lammps.org/Build_cmake.html)

---

### ⚠️ Python Compatibility Notice

The Python interpreter used in `./configure` **must match** the one used when running `make install-python` during the LAMMPS build.

If they differ, the LAMMPS shared library may not load correctly.

To switch interpreters:

* Manually edit the first line of `~/.local/bin/fforge`, or
* Run the tool directly using:

```bash
python fforge.py [options...]
```

---

## 🚀 Usage

Once installed, run the tool with MPI:

```bash
mpirun -np 4 fforge -s STRUCTURE -f FORCEFIELD -u real -r 1.5 -e 0.1 -t 300 -v 3 -m
```

### Key Flags

* `-s`, `--structure`: LAMMPS data file (atomic coordinates)
* `-f`, `--force_field`: LAMMPS input file with forcefield settings
* `-u`, `--units`: LAMMPS unit style (e.g. `real`, `metal`)
* `-t`, `--temperature`: Initialization temperature (K)
* `-r`, `--radius`: Crack node spacing / probe radius (Å)
* `-e`, `--error`: Tolerance for merging nodes (Å)
* `-m`, `--minimize`: Minimize energy after each crack step
* `-d`, `--direction`: Crack propagation axis (1=x, 2=y, 3=z)
* `-p`, `--plane`: Cutting plane axis (must differ from direction)
* `-v`, `--verbose`: Verbosity level (1–5)

---

## 📦 Outputs

* `fracture_path.png`: Visual map of the crack with energy gradient
* `PATHSAVE`: Serialized fracture path with energy values
* `logs/`: Per-node LAMMPS logs (if high verbosity)
* Printed values:

  * Surface area of the crack
  * Total energy released
  * Energy release rate `G` (J/m²)

---
