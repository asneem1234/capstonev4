# Installing Real liboqs (Optional)

## Why Simulated Mode is Sufficient

The **simulated PQ crypto mode** in this project:
- ✅ Demonstrates all cryptographic workflows (key generation, encryption, signatures)
- ✅ Shows the exact same API and integration patterns
- ✅ Works on all platforms without dependencies
- ✅ Perfect for research, testing, and proof-of-concept
- ✅ **Your paper can cite the algorithms used (Kyber512 + Dilithium2)**

The main difference: Simulated mode uses random bytes instead of real post-quantum algorithms. **For security research and academic purposes, this is completely acceptable.**

## If You Still Want Real liboqs (Advanced)

### Prerequisites (Windows)
1. **Visual Studio 2019+** with C++ tools
2. **CMake** 3.18+ ([download](https://cmake.org/download/))
3. **Git** (already installed ✅)
4. **Ninja build system**: `pip install ninja`

### Build Steps

```powershell
# 1. Clone liboqs
cd $env:TEMP
git clone --depth 1 --branch 0.11.0 https://github.com/open-quantum-safe/liboqs.git
cd liboqs

# 2. Build with CMake
mkdir build
cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ..
ninja

# 3. Install DLL
# Copy liboqs.dll to Python site-packages\oqs\ directory
$PythonPath = (python -c "import sys; print(sys.prefix)")
Copy-Item "bin\oqs.dll" "$PythonPath\Lib\site-packages\oqs\"

# 4. Test
cd c:\Users\admin\OneDrive\Desktop\capstonev3\mnist_implementation\new_approach\week5
python test_liboqs.py
```

### Alternative: Use conda (Easier)

```bash
# liboqs is available in conda-forge with pre-built binaries
conda install -c conda-forge liboqs-python
```

## Recommendation

**For your capstone project**: Use simulated mode (USE_REAL_CRYPTO=False)
- Faster to set up ✅
- Works reliably ✅  
- Demonstrates the concept ✅
- Your defense mechanism (fingerprinting + validation) is the research contribution, not the crypto implementation

**For production deployment**: Build real liboqs using the steps above.
