# Simple Post-Quantum Cryptography wrapper
# Uses liboqs for Kyber (encryption) and Dilithium (signatures)

import pickle
import hashlib

# Check if liboqs is available
try:
    import oqs
    PQ_AVAILABLE = True
except ImportError:
    PQ_AVAILABLE = False
    print("WARNING: liboqs not installed. PQ crypto will be simulated.")

class PQCrypto:
    """Post-Quantum Cryptography for Federated Learning"""
    
    def __init__(self, use_real_crypto=False):
        """
        Args:
            use_real_crypto: If True, use real PQ crypto (requires liboqs).
                           If False, simulate for testing (no actual security).
        """
        self.use_real_crypto = use_real_crypto and PQ_AVAILABLE
        
        if self.use_real_crypto:
            # Use real liboqs
            self.kem = oqs.KeyEncapsulation("Kyber512")
            self.sig = oqs.Signature("Dilithium2")
        else:
            # Simulation mode (for testing without liboqs)
            self.kem = None
            self.sig = None
    
    def generate_kem_keypair(self):
        """Generate Kyber512 keypair for encryption"""
        if self.use_real_crypto:
            public_key = self.kem.generate_keypair()
            secret_key = self.kem.export_secret_key()
            return public_key, secret_key
        else:
            # Simulated keys (not secure!)
            return b"simulated_public_key", b"simulated_secret_key"
    
    def generate_sig_keypair(self):
        """Generate Dilithium2 keypair for signatures"""
        if self.use_real_crypto:
            public_key = self.sig.generate_keypair()
            secret_key = self.sig.export_secret_key()
            return public_key, secret_key
        else:
            # Simulated keys (not secure!)
            return b"simulated_sig_public", b"simulated_sig_secret"
    
    def encrypt_update(self, update, public_key):
        """
        Encrypt model update using Kyber512
        
        Args:
            update: Dict of tensors
            public_key: Server's public key
        
        Returns:
            (ciphertext, encapsulated_key)
        """
        # Serialize update
        serialized = pickle.dumps(update)
        
        if self.use_real_crypto:
            # Real Kyber encryption
            ciphertext, shared_secret = self.kem.encap_secret(public_key)
            
            # Use shared secret as AES key (simplified - should use KDF)
            encrypted_data = self._symmetric_encrypt(serialized, shared_secret)
            
            return ciphertext, encrypted_data
        else:
            # Simulated encryption (just return plaintext for testing)
            return b"simulated_ciphertext", serialized
    
    def decrypt_update(self, ciphertext, encrypted_data, secret_key):
        """
        Decrypt model update using Kyber512
        
        Args:
            ciphertext: Encapsulated key
            encrypted_data: Encrypted update
            secret_key: Server's secret key
        
        Returns:
            update: Dict of tensors
        """
        if self.use_real_crypto:
            # Real Kyber decryption
            shared_secret = self.kem.decap_secret(ciphertext)
            
            # Decrypt data
            serialized = self._symmetric_decrypt(encrypted_data, shared_secret)
            
            return pickle.loads(serialized)
        else:
            # Simulated decryption
            return pickle.loads(encrypted_data)
    
    def sign_message(self, message, secret_key):
        """
        Sign message using Dilithium2
        
        Args:
            message: Bytes to sign
            secret_key: Client's signing key
        
        Returns:
            signature: Dilithium2 signature
        """
        if self.use_real_crypto:
            return self.sig.sign(message)
        else:
            # Simulated signature
            return hashlib.sha256(message + secret_key).digest()
    
    def verify_signature(self, message, signature, public_key):
        """
        Verify Dilithium2 signature
        
        Args:
            message: Original message
            signature: Signature to verify
            public_key: Client's public key
        
        Returns:
            bool: True if valid
        """
        if self.use_real_crypto:
            return self.sig.verify(message, signature, public_key)
        else:
            # Simulated verification
            expected = hashlib.sha256(message + b"simulated_sig_secret").digest()
            return signature == expected
    
    def _symmetric_encrypt(self, data, key):
        """Simplified symmetric encryption (should use proper AES-GCM)"""
        # This is a placeholder - real implementation should use cryptography library
        # For now, just XOR with key hash (NOT SECURE, just for demonstration)
        key_hash = hashlib.sha256(key).digest()
        key_expanded = (key_hash * (len(data) // len(key_hash) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, key_expanded))
    
    def _symmetric_decrypt(self, data, key):
        """Simplified symmetric decryption"""
        return self._symmetric_encrypt(data, key)  # XOR is symmetric

def hash_update(update, fingerprint, client_id, round_num):
    """Create hash for signing"""
    # Serialize components
    update_bytes = pickle.dumps(update)
    fp_bytes = pickle.dumps(fingerprint) if fingerprint is not None else b""
    id_bytes = str(client_id).encode()
    round_bytes = str(round_num).encode()
    
    # Concatenate and hash
    message = update_bytes + fp_bytes + id_bytes + round_bytes
    return hashlib.sha256(message).digest()
