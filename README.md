```bash

```markdown
# wdw.gc

**A high-performance C++ solver for the Wheeler-DeWitt Equation in the Early Universe using the Initial Value Problem (IVP) approach.**

## Overview

`wdw.gc` is a scientific computing tool designed to solve the Wheeler-DeWitt (2+1) equation. It utilizes the **Crank-Nicolson method** to discretize the system, generating a large sparse linear system, which is then efficiently solved at each time iteration using the **Conjugate Gradient (CG) method**.

This software is optimized for High-Performance Computing (HPC) environments, heavily utilizing **OpenMP** for parallel execution to handle large multidimensional grids.

## Repository Structure

To keep the project organized and maintainable, the files are structured as follows:
* `src/`: C++ source files (`.cpp`).
* `include/`: C++ header files (`.hpp`).
* `docs/`: Technical documentation and manuals (`wdw.gc.manual.pdf`).
* `examples/`: Configuration files and test cases (`wdw.ini`, `fast.ini`).

## Prerequisites

To build and run this project, you will need:
* A C++ compiler with **C++17** support (e.g., GCC 7+, Clang 5+, MSVC).
* **OpenMP** (for multithreading/parallelism).
* **CMake** (version 3.10 or higher).

---

## Quick Start / Auto-Runner (Windows)

For a seamless "plug and play" experience on Windows, you can use the provided automation script. You do not need to manually configure CMake or use the terminal.

1. Edit your physical parameters in `examples/fast.ini` or `examples/wdw.ini`.
2. **Double-click** the `run.bat` file located in the root directory.

The script will automatically set up the build environment, compile the code in the background, and start the simulation. 

> **Tip:** You can also **drag and drop** any `.ini` file directly onto the `run.bat` icon to instantly run that specific configuration!

---

## Manual Build (Linux / macOS / HPC)

If you are on Linux, macOS, or prefer full control in an HPC environment, you can build the project manually using CMake. Open your terminal and run the following commands from the root directory of this repository:

```bash
# 1. Create a build directory and navigate into it
mkdir build
cd build

# 2. Generate the build system files
cmake ..

# 3. Compile the code (using all available CPU cores in Release mode)
cmake --build . --config Release -j

```

After a successful compilation, the `wdw.gc` executable will be generated.
*(Note: On Windows, CMake usually places the executable inside a `Release/` subdirectory).*

## Manual Execution (Terminal)

The execution is parameterized by an INI configuration file.

From inside the `build/` directory, run:

**On Linux/macOS:**

```bash
./wdw.gc ../examples/wdw.ini

```

**On Windows (PowerShell or CMD):**

```powershell
.\Release\wdw.gc.exe ..\examples\wdw.ini

```

*(Note: If the `<ini.file>` argument is omitted, the program will look for `wdw.ini` in the current working directory).*

### Output

* **Standard Output (Console):** Prints average values for several parameters during execution.
* **Standard Error (Console):** Logs the Conjugate Gradient (CG) iteration convergence at each time step.

## License

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License v3.0** as published by the Free Software Foundation. See the `LICENSE` file for more details.

```

```
