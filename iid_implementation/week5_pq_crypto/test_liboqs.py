# Test liboqs installation
import sys
import os

print("Python version:", sys.version)
print("Python prefix:", sys.prefix)

try:
    import oqs
    print("✓ oqs module imported successfully")
    try:
        print("liboqs version:", oqs.oqs_version())
        print("✓ liboqs C library loaded successfully!")
        
        # Test KEM
        print("\nTesting Kyber512...")
        kem = oqs.KeyEncapsulation("Kyber512")
        public_key = kem.generate_keypair()
        print(f"✓ Kyber512 keypair generated! Public key size: {len(public_key)} bytes")
        
        # Test Signature
        print("\nTesting Dilithium2...")
        sig = oqs.Signature("Dilithium2")
        sig_public_key = sig.generate_keypair()
        print(f"✓ Dilithium2 keypair generated! Public key size: {len(sig_public_key)} bytes")
        
        print("\n✅ ALL TESTS PASSED! Real PQ crypto is working!")
        
    except Exception as e:
        print(f"✗ Error loading liboqs C library: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"✗ Failed to import oqs module: {e}")
